import { io } from "socket.io-client";

const URL = "http://kykint.com:5980/dashboard";

export const socket = io(URL, { transports: ["websocket"] });
