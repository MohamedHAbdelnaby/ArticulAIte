from flask import Flask, request, jsonify
import traceback
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import openai


app = Flask(__name__)
app.debug = True  


# =============================
# 2. Initialize the GPT-4 model via LangChain
# =============================
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-4",
    temperature=0.5,
    max_tokens=512
)

# =============================
# 3. Initialize the NVIDIA USD-code client
# =============================
nvidia_client = openai.OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# =============================
# 4. Define an endpoint to generate an office scene
# =============================
@app.route("/generate_office_scene", methods=["POST"])
def generate_office_scene():
    try:
        data = request.get_json() or {}
        prompt = data.get(
            "prompt",
            (
                "Generate a detailed 3D scene description for a modern office environment. "
                "Include photorealistic textures, realistic lighting, advanced materials, and "
                "functional furniture suitable for a virtual technical job interview setting."
            )
        )
        
        # Step 1: Generate the scene description using GPT-4.
        messages = [
            SystemMessage(content="You are an expert in generating detailed 3D scene descriptions for virtual environments."),
            HumanMessage(content=prompt)
        ]
        scene_description_obj = llm(messages)
        scene_description = str(scene_description_obj)
        
        # Step 2: Prepare a prompt for NVIDIA’s USD-code model.
        usd_prompt = (
            "Generate USD-Python code that creates a 3D scene in NVIDIA Omniverse for the following office scene description:\n\n"
            f"{scene_description}\n\n"
            "The code should include creation of assets, lighting, materials, and camera settings appropriate for a modern office environment."
        )
        
        # Call NVIDIA’s USD-code model with streaming response.
        completion = nvidia_client.chat.completions.create(
            model="nvidia/usdcode-llama-3.1-70b-instruct",
            messages=[{"role": "user", "content": usd_prompt}],
            temperature=0.1,
            top_p=1,
            max_tokens=1024,
            extra_body={"expert_type": "auto"},
            stream=True
        )
        
        usd_code = ""
        for chunk in completion:
            delta = chunk.choices[0].delta
            # Access the content attribute directly instead of using .get()
            if delta and delta.content is not None:
                usd_code += delta.content
        
        # Return both the scene description and the USD code.
        return jsonify({
            "scene_description": scene_description,
            "usd_code": usd_code
        })
    except Exception as e:
        print("Error occurred:\n", traceback.format_exc())
        return jsonify({"error": "Internal server error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)