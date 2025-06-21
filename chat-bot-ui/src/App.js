// src/App.js
import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import ChatBox from "./ChatBox";
import "./App.css";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<ChatBox />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
