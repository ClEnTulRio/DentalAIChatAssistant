document.addEventListener('DOMContentLoaded', function() {
    const chatContainer = document.getElementById('chat-container');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');

    function addMessage(content, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.textContent = content;
        
        messageDiv.appendChild(messageContent);
        chatContainer.appendChild(messageDiv);
        
        // Scroll to bottom
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function handlePlaceholders(response) {
        if (response.info_cards && response.info_cards.length > 0) {
            console.log('Info Cards:', response.info_cards);
        }
        if (response.models && response.models.length > 0) {
            console.log('3D Models:', response.models);
        }
    }

    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        // Disable input and button while processing
        userInput.disabled = true;
        sendButton.disabled = true;

        // Add user message to chat
        addMessage(message, 'user');
        userInput.value = '';

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            });

            const data = await response.json();
            
            if (response.ok) {
                // Add AI response to chat
                addMessage(data.response, 'assistant');
                handlePlaceholders(data);
            } else {
                addMessage('Sorry, there was an error processing your request.', 'error');
            }
        } catch (error) {
            addMessage('Sorry, there was an error connecting to the server.', 'error');
            console.error('Error:', error);
        }

        // Re-enable input and button
        userInput.disabled = false;
        sendButton.disabled = false;
        userInput.focus();
    }

    // Event listeners
    sendButton.addEventListener('click', sendMessage);
    
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    // Focus input on load
    userInput.focus();
});
