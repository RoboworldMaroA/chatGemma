# Marek Augustyn
# 02 August 2025
# It is working flask app with gemma 2b finetuned model
# It is running on port 5001
#To run local server use command: python app.py
# To run on production use program app_gunicorn.py

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import AutoTokenizer # Only need tokenizer from transformers
from peft import AutoPeftModelForCausalLM # Import this for loading PEFT models
import torch
import os # Import os for path handling

# Determine the device globally
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}") # This should print 'Using device: mps'
# Define the absolute path to your frontend folder (assuming a specific structure)
# If app.py is in 'my_gemma_chat_app_backend/' and frontend is in 'my_gemma_chat_app_frontend/'
# Corrected static_folder path for a separate frontend project
basedir = os.path.abspath(os.path.dirname(__file__))
frontend_folder_path = os.path.join(basedir, '', 'my_gemma_chat_app_frontend') # Adjust 'my_gemma_chat_app_frontend' if your actual folder name is different

app = Flask(__name__, static_folder=frontend_folder_path) # Point static_folder to the frontend root
CORS(app)

MODEL_PATH = "gemma_2b_finetuned_v2/final_checkpoint" # Adjust as per your exact structure

tokenizer = None
model = None

def load_model():
    global tokenizer, model #, text_generator
    print(f"Loading model from {MODEL_PATH}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Load the fine-tuned model with its PEFT adapters
    model = AutoPeftModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16, # Match your training dtype
        device_map="auto",
    )
    model.to(device)  # Ensure the model is on the correct device
    print(f"Model loaded to device: {device}")
    model.config.use_cache = True
    model.eval() # Set model to evaluation mode
    print("Model loaded successfully!")

with app.app_context():
    load_model()

# ... rest of your app.py
# Route to serve the main HTML file (your frontend)
@app.route('/')
def serve_index():
    # Make sure 'index.html' is directly in your static_folder (my_gemma_chat_app_frontend)
    return send_from_directory(app.static_folder, 'index.html')

# This route serves any other static files (CSS, JS, images) from the frontend folder


# For example, a request to /style.css will serve frontend_folder_path/style.css
@app.route('/<path:filename>')
def serve_static_files(filename):
    # Security: Ensure the requested file is within the static directory.
    # send_from_directory handles this mostly, but good to be explicit.
    if os.path.exists(os.path.join(app.static_folder, filename)):
        return send_from_directory(app.static_folder, filename)
    else:
        # For a Single Page Application, you might want to redirect to index.html
        # for client-side routing instead of returning a 404.
        return "File not found", 404


@app.route("/chat", methods=["POST"])
def chat():
    user_question = request.json.get("question")
    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    prompt = f"<bos><start_of_turn>user\n{user_question}<end_of_turn>\n<start_of_turn>model\n"

    try:
        # Tokenize the input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device) # Ensure inputs are on the correct device

        # Generate response
        #do_sample=True,
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id # Important for generation
                # eos_token_id=tokenizer.eos_token_id # Explicitly pass it
            )

        # Decode only the newly generated part
        # The output will contain the prompt + generated text.
        # We need to slice it to get only the new tokens.
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0, input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)

        # Post-process for chat template artifacts
        if "<end_of_turn>" in generated_text:
            generated_text = generated_text.split("<end_of_turn>")[0].strip()

        # If skip_special_tokens=True in decode, you might not need this line
        # but it's good practice to keep the prompt format in mind.
        if "<start_of_turn>user" in generated_text:
             generated_text = generated_text.split("<start_of_turn>user")[0].strip()


        return jsonify({"response": generated_text})

    except Exception as e:
        print(f"Error during inference: {e}")
        return jsonify({"error": "An error occurred during response generation."}), 500

if __name__ == "__main__":
    # Make sure to set debug=True only for development, not production
    # app.run(debug=True, port=5000)
    app.run(host='0.0.0.0', debug=True, port=5001)

