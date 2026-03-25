package agnsdk

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const jsonRPCVersion = "2.0"

type Options struct {
	BinaryPath string
	Args       []string
	Env        []string
}

type Client struct {
	cmd      *exec.Cmd
	stdin    io.WriteCloser
	stdout   io.ReadCloser
	mu       sync.Mutex
	writeMu  sync.Mutex
	pending  map[string]chan rpcResponse
	handlers map[string]func(Event)
	nextID   int64
	done     chan struct{}
}

type TurnParams struct {
	Prompt         string `json:"prompt"`
	ThreadID       string `json:"thread_id,omitempty"`
	RestrictOutput bool   `json:"restrict_output,omitempty"`
	Stream         bool   `json:"stream,omitempty"`
}

type TurnResult struct {
	ThreadID string `json:"thread_id"`
	Response string `json:"response"`
}

type ThreadSummary struct {
	ID        string    `json:"thread_id"`
	Preview   string    `json:"preview"`
	UpdatedAt time.Time `json:"updated_at"`
}

type ThreadReadResult struct {
	ThreadID  string              `json:"thread_id"`
	UpdatedAt time.Time           `json:"updated_at"`
	Messages  []ThreadReadMessage `json:"messages"`
}

type ThreadReadMessage struct {
	ID         string          `json:"id"`
	CreatedAt  time.Time       `json:"created_at"`
	Role       string          `json:"role"`
	Kind       string          `json:"kind"`
	Text       string          `json:"text,omitempty"`
	ToolCalls  json.RawMessage `json:"tool_calls,omitempty"`
	ToolOutput json.RawMessage `json:"tool_output,omitempty"`
}

type ThreadListParams struct {
	Limit  int    `json:"limit,omitempty"`
	Cursor string `json:"cursor,omitempty"`
}

type ThreadListResult struct {
	Data       []ThreadSummary `json:"data"`
	NextCursor *string         `json:"next_cursor"`
}

type ThreadReadParams struct {
	ThreadID string `json:"thread_id"`
}

type CancelParams struct {
	RequestID string `json:"request_id"`
}

type Event struct {
	Method     string `json:"method,omitempty"`
	RequestID  string `json:"request_id,omitempty"`
	Type       string `json:"type,omitempty"`
	ThreadID   string `json:"thread_id,omitempty"`
	TurnID     string `json:"turn_id,omitempty"`
	ItemID     string `json:"item_id,omitempty"`
	Delta      string `json:"delta,omitempty"`
	ToolName   string `json:"tool_name,omitempty"`
	ToolCallID string `json:"tool_call_id,omitempty"`
}

type agentMessageDeltaNotification struct {
	RequestID string `json:"request_id"`
	ThreadID  string `json:"thread_id"`
	TurnID    string `json:"turn_id"`
	ItemID    string `json:"item_id"`
	Delta     string `json:"delta"`
}

type itemLifecycleNotification struct {
	RequestID string `json:"request_id"`
	ThreadID  string `json:"thread_id"`
	TurnID    string `json:"turn_id"`
	ItemID    string `json:"item_id"`
	ToolName  string `json:"tool_name"`
}

type turnLifecycleNotification struct {
	RequestID string `json:"request_id"`
	ThreadID  string `json:"thread_id"`
	TurnID    string `json:"turn_id"`
}

type rpcRequest struct {
	JSONRPC string      `json:"jsonrpc"`
	ID      int64       `json:"id"`
	Method  string      `json:"method"`
	Params  interface{} `json:"params,omitempty"`
}

type rpcResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *rpcError       `json:"error,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

func Start(ctx context.Context, opts Options) (*Client, error) {
	binary := strings.TrimSpace(opts.BinaryPath)
	if binary == "" {
		binary = "agn"
	}
	args := opts.Args
	if len(args) == 0 {
		args = []string{"serve"}
	}

	cmd := exec.CommandContext(ctx, binary, args...)
	if len(opts.Env) > 0 {
		cmd.Env = append(os.Environ(), opts.Env...)
	}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("open stdin: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("open stdout: %w", err)
	}
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start agn: %w", err)
	}

	client := &Client{
		cmd:      cmd,
		stdin:    stdin,
		stdout:   stdout,
		pending:  make(map[string]chan rpcResponse),
		handlers: make(map[string]func(Event)),
		done:     make(chan struct{}),
	}
	go client.readLoop()
	return client, nil
}

func (c *Client) Turn(ctx context.Context, params TurnParams, onEvent func(Event)) (TurnResult, error) {
	return c.runTurn(ctx, "turn/start", params, onEvent)
}

func (c *Client) Resume(ctx context.Context, threadID string, params TurnParams, onEvent func(Event)) (TurnResult, error) {
	if strings.TrimSpace(threadID) == "" {
		return TurnResult{}, errors.New("thread ID is required")
	}
	params.ThreadID = threadID
	return c.runTurn(ctx, "thread/resume", params, onEvent)
}

func (c *Client) runTurn(ctx context.Context, method string, params TurnParams, onEvent func(Event)) (TurnResult, error) {
	if strings.TrimSpace(params.Prompt) == "" {
		return TurnResult{}, errors.New("prompt is required")
	}
	requestID := atomic.AddInt64(&c.nextID, 1)
	idKey := strconv.FormatInt(requestID, 10)
	respCh := make(chan rpcResponse, 1)
	defer close(respCh)

	c.mu.Lock()
	c.pending[idKey] = respCh
	if onEvent != nil {
		c.handlers[idKey] = onEvent
	}
	c.mu.Unlock()

	if err := c.sendRequest(rpcRequest{JSONRPC: jsonRPCVersion, ID: requestID, Method: method, Params: params}); err != nil {
		return TurnResult{}, err
	}

	select {
	case <-ctx.Done():
		return TurnResult{}, ctx.Err()
	case resp := <-respCh:
		if resp.Error != nil {
			return TurnResult{}, fmt.Errorf("rpc error %d: %s", resp.Error.Code, resp.Error.Message)
		}
		var result TurnResult
		if err := json.Unmarshal(resp.Result, &result); err != nil {
			return TurnResult{}, fmt.Errorf("parse result: %w", err)
		}
		return result, nil
	}
}

func (c *Client) ThreadList(ctx context.Context, limit int) (ThreadListResult, error) {
	requestID := atomic.AddInt64(&c.nextID, 1)
	idKey := strconv.FormatInt(requestID, 10)
	respCh := make(chan rpcResponse, 1)
	defer close(respCh)

	c.mu.Lock()
	c.pending[idKey] = respCh
	c.mu.Unlock()

	params := ThreadListParams{Limit: limit}
	if err := c.sendRequest(rpcRequest{JSONRPC: jsonRPCVersion, ID: requestID, Method: "thread/list", Params: params}); err != nil {
		return ThreadListResult{}, err
	}

	select {
	case <-ctx.Done():
		return ThreadListResult{}, ctx.Err()
	case resp := <-respCh:
		if resp.Error != nil {
			return ThreadListResult{}, fmt.Errorf("rpc error %d: %s", resp.Error.Code, resp.Error.Message)
		}
		var result ThreadListResult
		if err := json.Unmarshal(resp.Result, &result); err != nil {
			return ThreadListResult{}, fmt.Errorf("parse result: %w", err)
		}
		return result, nil
	}
}

func (c *Client) ThreadRead(ctx context.Context, threadID string) (ThreadReadResult, error) {
	if strings.TrimSpace(threadID) == "" {
		return ThreadReadResult{}, errors.New("thread ID is required")
	}
	requestID := atomic.AddInt64(&c.nextID, 1)
	idKey := strconv.FormatInt(requestID, 10)
	respCh := make(chan rpcResponse, 1)
	defer close(respCh)

	c.mu.Lock()
	c.pending[idKey] = respCh
	c.mu.Unlock()

	params := ThreadReadParams{ThreadID: threadID}
	if err := c.sendRequest(rpcRequest{JSONRPC: jsonRPCVersion, ID: requestID, Method: "thread/read", Params: params}); err != nil {
		return ThreadReadResult{}, err
	}

	select {
	case <-ctx.Done():
		return ThreadReadResult{}, ctx.Err()
	case resp := <-respCh:
		if resp.Error != nil {
			return ThreadReadResult{}, fmt.Errorf("rpc error %d: %s", resp.Error.Code, resp.Error.Message)
		}
		var result ThreadReadResult
		if err := json.Unmarshal(resp.Result, &result); err != nil {
			return ThreadReadResult{}, fmt.Errorf("parse result: %w", err)
		}
		return result, nil
	}
}

func (c *Client) Cancel(ctx context.Context, requestID string) error {
	if strings.TrimSpace(requestID) == "" {
		return errors.New("request ID is required")
	}
	respCh := make(chan rpcResponse, 1)
	defer close(respCh)
	rpcID := atomic.AddInt64(&c.nextID, 1)
	idKey := strconv.FormatInt(rpcID, 10)
	params := CancelParams{RequestID: requestID}

	c.mu.Lock()
	c.pending[idKey] = respCh
	c.mu.Unlock()

	if err := c.sendRequest(rpcRequest{JSONRPC: jsonRPCVersion, ID: rpcID, Method: "turn/interrupt", Params: params}); err != nil {
		return err
	}

	select {
	case <-ctx.Done():
		return ctx.Err()
	case resp := <-respCh:
		if resp.Error != nil {
			return fmt.Errorf("rpc error %d: %s", resp.Error.Code, resp.Error.Message)
		}
		return nil
	}
}

func (c *Client) Close() error {
	close(c.done)
	if c.stdin != nil {
		_ = c.stdin.Close()
	}
	if c.cmd != nil && c.cmd.Process != nil {
		_ = c.cmd.Process.Kill()
		_ = c.cmd.Wait()
	}
	return nil
}

func (c *Client) sendRequest(req rpcRequest) error {
	payload, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("marshal request: %w", err)
	}
	payload = append(payload, '\n')
	c.writeMu.Lock()
	_, err = c.stdin.Write(payload)
	c.writeMu.Unlock()
	if err != nil {
		return fmt.Errorf("write request: %w", err)
	}
	return nil
}

func (c *Client) readLoop() {
	scanner := bufio.NewScanner(c.stdout)
	scanner.Buffer(make([]byte, 0, 64*1024), 10*1024*1024)
	for scanner.Scan() {
		select {
		case <-c.done:
			return
		default:
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var resp rpcResponse
		if err := json.Unmarshal([]byte(line), &resp); err != nil {
			fmt.Fprintf(os.Stderr, "agn-sdk: failed to parse response: %v\n", err)
			continue
		}
		if resp.Method != "" && len(resp.ID) == 0 {
			c.handleNotification(resp)
			continue
		}
		idKey := strings.TrimSpace(string(resp.ID))
		if idKey == "" {
			continue
		}
		c.mu.Lock()
		ch := c.pending[idKey]
		delete(c.pending, idKey)
		delete(c.handlers, idKey)
		c.mu.Unlock()
		if ch != nil {
			ch <- resp
		}
	}
}

func (c *Client) handleNotification(resp rpcResponse) {
	switch resp.Method {
	case "item/agentMessage/delta":
		var payload agentMessageDeltaNotification
		if err := json.Unmarshal(resp.Params, &payload); err != nil {
			return
		}
		c.dispatchEvent(Event{
			Method:    resp.Method,
			RequestID: payload.RequestID,
			ThreadID:  payload.ThreadID,
			TurnID:    payload.TurnID,
			ItemID:    payload.ItemID,
			Delta:     payload.Delta,
		})
	case "item/started", "item/completed":
		var payload itemLifecycleNotification
		if err := json.Unmarshal(resp.Params, &payload); err != nil {
			return
		}
		c.dispatchEvent(Event{
			Method:    resp.Method,
			RequestID: payload.RequestID,
			ThreadID:  payload.ThreadID,
			TurnID:    payload.TurnID,
			ItemID:    payload.ItemID,
			ToolName:  payload.ToolName,
		})
	case "turn/started", "turn/completed":
		var payload turnLifecycleNotification
		if err := json.Unmarshal(resp.Params, &payload); err != nil {
			return
		}
		c.dispatchEvent(Event{
			Method:    resp.Method,
			RequestID: payload.RequestID,
			ThreadID:  payload.ThreadID,
			TurnID:    payload.TurnID,
		})
	}
}

func (c *Client) dispatchEvent(event Event) {
	requestID := strings.TrimSpace(event.RequestID)
	c.mu.Lock()
	if requestID != "" {
		handler := c.handlers[requestID]
		c.mu.Unlock()
		if handler != nil {
			handler(event)
		}
		return
	}
	handlers := make([]func(Event), 0, len(c.handlers))
	for _, handler := range c.handlers {
		handlers = append(handlers, handler)
	}
	c.mu.Unlock()
	for _, handler := range handlers {
		handler(event)
	}
}
