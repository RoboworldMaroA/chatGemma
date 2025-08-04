const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatMessages = document.getElementById("chat-messages");

chatForm.addEventListener("submit", async (e) => {
    e.preventDefault(); // Prevent default form submission (page reload)

    const userQuestion = chatInput.value.trim();
    if (userQuestion === "") {
        return; // Don't send empty messages
    }

    // Display user's message
    appendMessage("user", userQuestion);
    chatInput.value = ""; // Clear input field

    // Scroll to the bottom
    scrollToBottom();

    // Show a loading indicator
    const loadingMessage = appendMessage("bot", "Thinking...");
    scrollToBottom();

    try {
        // Server Host for flask and gunicorn
        const response = await fetch("http://86.46.205.2:5001/chat", {
        // Local Host for flask and gunicorn
        // const response = await fetch("http://192.168.1.10:5001/chat", {
        // const response = await fetch("http://86.46.205.2:5001/chat", {
        // const response = await fetch("http://127.0.0.1:5000/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ question: userQuestion }),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || "Network response was not ok.");
        }

        const data = await response.json();

        // Update the loading message with the actual response
        loadingMessage.textContent = data.response;

    } catch (error) {
        console.error("Error:", error);
        loadingMessage.textContent = "Sorry, I couldn't get a response. Please try again.";
    } finally {
        scrollToBottom(); // Ensure scroll to bottom after response
    }
});

function appendMessage(sender, text) {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message", sender);
    messageDiv.textContent = text;
    chatMessages.appendChild(messageDiv);
    return messageDiv; // Return the new element for later updates (e.g., loading message)
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}