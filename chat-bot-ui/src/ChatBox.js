// src/ChatBox.js
import React, { useState } from "react";
import axios from "axios";

const ChatBox = () => {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { sender: "You", text: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");

    try {
      const res = await axios.post("http://localhost:8000/chat", {
        text: input,
      });
      const botMessage = { sender: "Bot", text: res.data.reply };
      setMessages((prev) => [...prev, botMessage]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { sender: "Bot", text: "Server error. Try again later." },
      ]);
      console.error(err);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-box">
        <div className="chat-messages">
          {messages.map((msg, i) => (
            <div key={i} className={`chat-message ${msg.sender === "You" ? "user" : "bot"}`}>
              <span className="chat-sender">{msg.sender}:</span>
              <span>{msg.text}</span>
            </div>
          ))}
        </div>
        <form onSubmit={sendMessage} className="chat-input-form">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            className="chat-input"
          />
          <button type="submit" className="chat-send">Send</button>
        </form>
      </div>
    </div>
  );
};

export default ChatBox;
