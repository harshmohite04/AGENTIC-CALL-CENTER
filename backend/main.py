from langchain import OpenAI, LLMChain
from langchain.memory import MongoDBChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.tools import BaseTool
import requests
import time


# 🌟 MongoDB Chat Memory
mongo_memory = MongoDBChatMessageHistory(
    connection_string="mongodb+srv://<username>:<password>@cluster0.mongodb.net",
    session_id="user-session-id",
    database_name="callCenterDB",
    collection_name="leads"
)

# 🌟 Step 1: Get user details
user_details = {
    "name": "Akash",
    "age": 23,
    "phone": "+919812345678"
}

# 🌟 Step 2: Define Tools
# ---------------------------------------

# 🎤 ElevenLabs: Text to Speech
def get_voice(text: str) -> str:
    response = requests.post(
        "https://api.elevenlabs.io/v1/text-to-speech",
        headers={"Authorization": "Bearer <ELEVENLABS_API_KEY>"},
        json={"text": text, "voice": "Raju"}
    )
    audio_url = response.json()["audio_url"]
    return audio_url


# 📞 Twilio: Place call and record
def make_call(phone: str, audio_url: str) -> str:
    response = requests.post(
        "https://api.twilio.com/2010-04-01/Accounts/<ACCOUNT_SID>/Calls.json",
        data={
            "To": phone,
            "From": "+19129136422",
            "Url": audio_url,
            "Record": True
        },
        auth=("<ACCOUNT_SID>", "<TWILIO_AUTH_TOKEN>")
    )
    call_sid = response.json()["sid"]
    print(f"📞 Call initiated: {call_sid}")

    # Wait for recording to process
    time.sleep(10)

    # Get recording URL
    recordings_resp = requests.get(
        f"https://api.twilio.com/2010-04-01/Accounts/<ACCOUNT_SID>/Calls/{call_sid}/Recordings.json",
        auth=("<ACCOUNT_SID>", "<TWILIO_AUTH_TOKEN>")
    )
    recording_url = recordings_resp.json()["recordings"][0]["media_url"]
    return recording_url


# 📝 Whisper: Transcribe Audio
def transcribe_audio(audio_url: str) -> str:
    audio_data = requests.get(audio_url).content
    response = requests.post(
        "https://api.openai.com/v1/audio/transcriptions",
        headers={"Authorization": "Bearer <OPENAI_API_KEY>"},
        files={"file": ("recording.wav", audio_data)},
        data={"model": "whisper-1"}
    )
    transcription = response.json()["text"]
    return transcription


# 🌟 LangChain Tool Wrappers
class GetVoiceTool(BaseTool):
    name = "get_voice"
    description = "Convert text to Hinglish voice using ElevenLabs."

    def _run(self, text: str):
        return get_voice(text)

class MakeCallTool(BaseTool):
    name = "make_call"
    description = "Place a call and record user response using Twilio."

    def _run(self, phone: str, audio_url: str):
        return make_call(phone, audio_url)

class TranscribeAudioTool(BaseTool):
    name = "transcribe_audio"
    description = "Transcribe audio recording using Whisper."

    def _run(self, audio_url: str):
        return transcribe_audio(audio_url)

# 🌟 LangChain Tools
tools = [
    GetVoiceTool(),
    MakeCallTool(),
    TranscribeAudioTool()
]

# 🌟 Step 3: LLM Chain for Lead Analysis
prompt = PromptTemplate(
    input_variables=["user_reply"],
    template="""
This is the user's reply: "{user_reply}"
Analyze the reply and decide:
- If they are interested → Output: GOOD
- If they are not interested → Output: BAD
""",
)

llm = OpenAI(model="gpt-4", temperature=0)
lead_chain = LLMChain(prompt=prompt, llm=llm)

# 🌟 Step 4: AI Agent Orchestration
agent = initialize_agent(
    tools=tools,
    llm=llm,
    verbose=True,
    memory=mongo_memory
)

# 🌟 Step 5: Orchestrate the flow
def run_flow():
    # Prepare Hinglish voice message
    voice_message = f"Namaste {user_details['name']} ji! Main LearnPro se bol raha hoon. Aapka education aur career goals ke liye ek special offer hai. Kya aap 1 minute baat karna pasand karenge? Haan ya naa bataiye."
    audio_url = get_voice(voice_message)

    # Place the call and get recording
    recording_url = make_call(user_details["phone"], audio_url)

    # Transcribe user response
    user_reply = transcribe_audio(recording_url)
    print(f"📝 User Reply: {user_reply}")

    # Analyze lead
    lead_status = lead_chain.run(user_reply)
    print(f"✅ Lead Status: {lead_status}")

    # Save lead in MongoDB
    mongo_memory.add_message("AI Agent", {
        "name": user_details["name"],
        "age": user_details["age"],
        "phone": user_details["phone"],
        "reply": user_reply,
        "lead_status": lead_status
    })

run_flow()
