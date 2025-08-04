from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import AutoTokenizer # Only need tokenizer from transformers
from peft import AutoPeftModelForCausalLM # Import this for loading PEFT models
import torch

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

MODEL_PATH = "gemma_2b_finetuned_v2/final_checkpoint" # Adjust as per your exact structure

tokenizer = None
model = None
# text_generator = None # No need for pipeline if you're doing manual generation

def load_model():
    global tokenizer, model #, text_generator
    print(f"Loading model from {MODEL_PATH}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Load the fine-tuned model with its PEFT adapters
    model = AutoPeftModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16, # Match your training dtype
        device_map="auto",
        # token=os.environ.get("HF_TOKEN") # Only if base model requires auth
    )
    model.eval() # Set model to evaluation mode

    # If you still want to use pipeline for simplicity, you can,
    # but sometimes direct generation is more flexible with PEFT models.
    # text_generator = pipeline(
    #     "text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     torch_dtype=torch.float16,
    #     device_map="auto"
    # )
    print("Model loaded successfully!")

with app.app_context():
    load_model()

# ... rest of your app.py

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
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=250,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id # Important for generation
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
    app.run(debug=True, port=5001)