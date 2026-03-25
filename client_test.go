package agnsdk

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestClientTurnValidation(t *testing.T) {
	client := &Client{}
	_, err := client.Turn(context.Background(), TurnParams{Prompt: " "}, nil)
	require.Error(t, err)
}

func TestClientCancelValidation(t *testing.T) {
	client := &Client{}
	err := client.Cancel(context.Background(), " ")
	require.Error(t, err)
}

func TestClientTurnRequest(t *testing.T) {
	client, reqReader, respWriter, cleanup := setupTestClient(t)
	defer cleanup()

	ctx := context.Background()
	var result TurnResult
	var err error
	done := make(chan struct{})
	go func() {
		result, err = client.Turn(ctx, TurnParams{Prompt: "hello"}, nil)
		close(done)
	}()

	req := readRequest(t, reqReader)
	require.Equal(t, jsonRPCVersion, req.JSONRPC)
	require.Equal(t, int64(1), req.ID)
	require.Equal(t, "turn/start", req.Method)
	params, ok := req.Params.(map[string]any)
	require.True(t, ok)
	require.Equal(t, "hello", params["prompt"])

	writeResponse(t, respWriter, req.ID, TurnResult{ThreadID: "thread-1", Response: "ok"}, nil)
	<-done
	require.NoError(t, err)
	require.Equal(t, TurnResult{ThreadID: "thread-1", Response: "ok"}, result)
}

func TestClientCancelRequest(t *testing.T) {
	client, reqReader, respWriter, cleanup := setupTestClient(t)
	defer cleanup()

	ctx := context.Background()
	var err error
	done := make(chan struct{})
	go func() {
		err = client.Cancel(ctx, "req-1")
		close(done)
	}()

	req := readRequest(t, reqReader)
	require.Equal(t, jsonRPCVersion, req.JSONRPC)
	require.Equal(t, int64(1), req.ID)
	require.Equal(t, "turn/interrupt", req.Method)
	params, ok := req.Params.(map[string]any)
	require.True(t, ok)
	require.Equal(t, "req-1", params["request_id"])

	writeResponse(t, respWriter, req.ID, map[string]bool{"cancelled": true}, nil)
	<-done
	require.NoError(t, err)
}

func setupTestClient(t *testing.T) (*Client, *bufio.Reader, io.WriteCloser, func()) {
	t.Helper()
	serverToClientR, serverToClientW := io.Pipe()
	clientToServerR, clientToServerW := io.Pipe()
	client := &Client{
		stdin:    clientToServerW,
		stdout:   serverToClientR,
		pending:  make(map[string]chan rpcResponse),
		handlers: make(map[string]func(Event)),
		done:     make(chan struct{}),
	}
	go client.readLoop()
	reader := bufio.NewReader(clientToServerR)
	cleanup := func() {
		_ = serverToClientW.Close()
		_ = clientToServerW.Close()
		_ = clientToServerR.Close()
		_ = serverToClientR.Close()
		_ = client.Close()
	}
	return client, reader, serverToClientW, cleanup
}

func readRequest(t *testing.T, reader *bufio.Reader) rpcRequest {
	t.Helper()
	line, err := reader.ReadString('\n')
	require.NoError(t, err)
	line = strings.TrimSpace(line)
	var req rpcRequest
	require.NoError(t, json.Unmarshal([]byte(line), &req))
	return req
}

func writeResponse(t *testing.T, writer io.Writer, id int64, result any, errObj *rpcError) {
	t.Helper()
	resp := rpcResponse{JSONRPC: jsonRPCVersion, ID: json.RawMessage(strconv.FormatInt(id, 10))}
	if errObj != nil {
		resp.Error = errObj
	} else {
		payload, err := json.Marshal(result)
		require.NoError(t, err)
		resp.Result = payload
	}
	data, err := json.Marshal(resp)
	require.NoError(t, err)
	data = append(data, '\n')
	_, err = writer.Write(data)
	require.NoError(t, err)
}
