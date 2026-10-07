import type { ClientMessage, ServerMessage } from "./protocol/protocol";

export type ConnectionState = "connecting" | "open" | "closed";

/** A WebSocket to the live server that reconnects with backoff. */
export class Connection {
  private socket: WebSocket | null = null;
  private retryDelayMs = 500;

  constructor(
    private readonly url: string,
    private readonly onMessage: (message: ServerMessage) => void,
    private readonly onState: (state: ConnectionState) => void,
  ) {
    this.connect();
  }

  /** Send a message; returns false (and drops it) while disconnected. */
  send(message: ClientMessage): boolean {
    if (this.socket?.readyState !== WebSocket.OPEN) return false;
    this.socket.send(JSON.stringify(message));
    return true;
  }

  private connect(): void {
    this.onState("connecting");
    const socket = new WebSocket(this.url);
    this.socket = socket;
    socket.addEventListener("open", () => {
      this.retryDelayMs = 500;
      this.onState("open");
    });
    socket.addEventListener("message", (event: MessageEvent<string>) => {
      this.onMessage(JSON.parse(event.data) as ServerMessage);
    });
    socket.addEventListener("close", () => {
      this.onState("closed");
      setTimeout(() => this.connect(), this.retryDelayMs);
      this.retryDelayMs = Math.min(this.retryDelayMs * 2, 5000);
    });
  }
}

/** The server's WebSocket URL, relative to the page's own origin. */
export function defaultSocketUrl(): string {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${location.host}/ws`;
}
