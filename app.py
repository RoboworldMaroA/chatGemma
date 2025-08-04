
#Marek Augustyn
#02 August 2025
# It is working flask app with gemma 2b finetuned model 
# It is running on port 5001
# When you use gunicorn then use command:
#gunicorn --bind 0.0.0.0:5001 app:app --workers 1 --timeout 120
# Program is slower then using flask run but it is more stable and is using MPS
# To run local server use command: python app.py change name app_flask.py to app.py

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import torch
import os # Import os for path handling

# Determine the device globally
#If you use mps - graphics card then is error
# If you use cpu then it works
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
print(f"Using device: {device}")

# Define the absolute path to your frontend folder (assuming a specific structure)
# If app.py is in 'my_gemma_chat_app_backend/' and frontend is in 'my_gemma_chat_app_frontend/'
# Corrected static_folder path for a separate frontend project
basedir = os.path.abspath(os.path.dirname(__file__))
frontend_folder_path = os.path.join(basedir, '', 'my_gemma_chat_app_frontend') # Adjust 'my_gemma_chat_app_frontend' if your actual folder name is different

app = Flask(__name__, static_folder=frontend_folder_path) # Point static_folder to the frontend root
CORS(app) # Initialize CORS for your app

MODEL_PATH = os.path.join(basedir, "gemma_2b_finetuned_v2", "final_checkpoint") # Use os.path.join for robust path
# If gemma_2b_finetuned_v2 is inside your backend project, this path is correct.
# Otherwise, adjust basedir to point to its parent or an absolute path.

tokenizer = None
model = None

def load_model():
    global tokenizer, model
    print(f"Loading model from {MODEL_PATH}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoPeftModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.to(device)
    print(f"Model loaded to device: {device}")
    model.config.use_cache = True
    model.eval()
    print("Model loaded successfully!")

# Load model when the application starts up
# This block runs when gunicorn imports your 'app' object
# No need for with app.app_context() here when running with Gunicorn,
# as the model should be loaded as part of the module's initialization
# if it's needed immediately upon app start.
# However, if you have specific Flask context dependent operations,
# then `with app.app_context():` is correct for those.
# For model loading, generally just putting it outside functions is fine or calling it directly.
# with app.app_context():
#     load_model() # <--- Move this directly after model/tokenizer globals

load_model() # <--- Move this directly after model/tokenizer globals

# --- Static File Serving (for when Flask serves the frontend) ---
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

# --- API Endpoint ---
@app.route("/chat", methods=["POST"])
def chat():
    user_question = request.json.get("question")
    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    prompt = f"<bos><start_of_turn>user\n{user_question}<end_of_turn>\n<start_of_turn>model\n"

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200, # Adjust as needed
                do_sample=True,   # Set to False for deterministic output, or True for creative
                temperature=0.7,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id
            )

        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0, input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)

        if "<end_of_turn>" in generated_text:
            generated_text = generated_text.split("<end_of_turn>")[0].strip()
        if "<start_of_turn>user" in generated_text:
             generated_text = generated_text.split("<start_of_turn>user")[0].strip()
          # --- Add this line to print the response to the terminal ---
        # print(f"Generated Bot Response: {generated_text}")
             print(f"Generated Bot Response: {generated_text.strip()}")  # Print the response to the terminal

        # Ensure generated_text is not empty or None (your existing check)
        if not generated_text:
            print("Warning: Model generated empty text!")
            return jsonify({"response": "I'm sorry, I couldn't generate a response."}), 200
        # Return the generated text as a JSON response
        return jsonify({"response": generated_text})

    except Exception as e:
        print(f"Error during inference: {e}")
        return jsonify({"error": "An error occurred during response generation."}), 500

# Remove the __main__ block if using Gunicorn for deployment.
# If you keep it, ensure app.run() is NOT called when Gunicorn is running.
# For local development without Gunicorn, you would still use this:
# if __name__ == "__main__":
#     # Debug mode is DANGEROUS for public-facing servers. Use only for local dev.
#     app.run(host='0.0.0.0', debug=True, port=5000)
