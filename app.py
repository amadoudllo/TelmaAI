import os
import json
import requests
import tempfile
from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse, Gather
import whisper
from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────
HF_TOKEN      = os.environ.get("HF_TOKEN", "")
HF_MODEL_URL  = "https://api-inference.huggingface.co/models/AmadouDiarouga/telma-mistral-telecom-gn"
ELEVENLABS_KEY = os.environ.get("ELEVENLABS_KEY", "")
VOICE_ID       = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

INSTRUCTION = (
    "Tu es TELMA, un assistant vocal IA pour les telecoms en Guinee. "
    "Tu aides les clients d'Orange Guinee, MTN et Celcom. "
    "Reponds en francais simple, court et direct, maximum 2 phrases. "
    "Cite toujours le code USSD ou le numero quand c est pertinent."
)

# Charger Whisper une seule fois au démarrage
print("Chargement Whisper...")
whisper_model = whisper.load_model("base")
print("Whisper pret !")

# ── Fonctions utilitaires ─────────────────────────────────────────

def transcrire_audio(audio_url: str) -> str:
    """Télécharge et transcrit l'audio Twilio avec Whisper."""
    headers = {}
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
    auth_token  = os.environ.get("TWILIO_AUTH_TOKEN", "")

    resp = requests.get(audio_url, auth=(account_sid, auth_token))
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(resp.content)
        tmp_path = f.name

    result = whisper_model.transcribe(tmp_path, language="fr")
    return result["text"].strip()


def appeler_llm(question: str) -> str:
    """Appelle le modèle TELMA via HF Inference API."""
    prompt = f"<s>[INST] {INSTRUCTION}\n\n{question} [/INST]"

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 150,
            "temperature": 0.3,
            "repetition_penalty": 1.1,
            "return_full_text": False,
        }
    }

    try:
        resp = requests.post(HF_MODEL_URL, headers=headers, json=payload, timeout=30)
        data = resp.json()
        if isinstance(data, list) and len(data) > 0:
            return data[0].get("generated_text", "").strip()
        return "Je n'ai pas pu traiter votre demande. Réessayez."
    except Exception as e:
        print(f"Erreur LLM: {e}")
        return "Service temporairement indisponible. Appelez le 600 pour Orange ou le 8800 pour MTN."


def synthetiser_voix(texte: str) -> bytes:
    """Convertit le texte en audio via ElevenLabs."""
    client = ElevenLabs(api_key=ELEVENLABS_KEY)
    audio = client.generate(
        text=texte,
        voice=Voice(
            voice_id=VOICE_ID,
            settings=VoiceSettings(stability=0.5, similarity_boost=0.75)
        ),
        model="eleven_multilingual_v2"
    )
    return b"".join(audio)


# ── Routes Twilio ─────────────────────────────────────────────────

@app.route("/voice/accueil", methods=["POST"])
def accueil():
    """Premier message quand quelqu'un appelle."""
    resp = VoiceResponse()
    gather = Gather(
        input="speech",
        action="/voice/traiter",
        method="POST",
        language="fr-FR",
        speech_timeout="auto",
        timeout=5,
    )
    gather.say(
        "Bonjour, je suis TELMA, votre assistant telecoms. "
        "Comment puis-je vous aider aujourd'hui ?",
        language="fr-FR",
        voice="alice",
    )
    resp.append(gather)
    resp.redirect("/voice/accueil")
    return Response(str(resp), mimetype="text/xml")


@app.route("/voice/traiter", methods=["POST"])
def traiter():
    """Traite la réponse vocale de l'appelant."""
    resp = VoiceResponse()

    # Récupérer ce que l'utilisateur a dit
    speech_result = request.form.get("SpeechResult", "")
    audio_url     = request.form.get("RecordingUrl", "")

    if speech_result:
        question = speech_result
    elif audio_url:
        question = transcrire_audio(audio_url)
    else:
        resp.say("Je n'ai pas compris. Pouvez-vous répéter ?", language="fr-FR")
        resp.redirect("/voice/accueil")
        return Response(str(resp), mimetype="text/xml")

    print(f"Question reçue : {question}")

    # Obtenir la réponse du LLM
    reponse_texte = appeler_llm(question)
    print(f"Réponse TELMA : {reponse_texte}")

    # Synthèse vocale
    try:
        audio_bytes = synthetiser_voix(reponse_texte)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, dir="/tmp") as f:
            f.write(audio_bytes)
            audio_path = f.name

        # Servir l'audio
        resp.play(f"{request.host_url}voice/audio/{os.path.basename(audio_path)}")
    except Exception as e:
        print(f"Erreur TTS: {e}")
        resp.say(reponse_texte, language="fr-FR", voice="alice")

    # Proposer une autre question
    gather = Gather(
        input="speech",
        action="/voice/traiter",
        method="POST",
        language="fr-FR",
        speech_timeout="auto",
        timeout=5,
    )
    gather.say("Avez-vous une autre question ?", language="fr-FR", voice="alice")
    resp.append(gather)
    resp.say("Merci d'avoir appelé TELMA. Au revoir !", language="fr-FR", voice="alice")

    return Response(str(resp), mimetype="text/xml")


@app.route("/voice/audio/<filename>")
def servir_audio(filename):
    """Sert les fichiers audio générés."""
    filepath = f"/tmp/{filename}"
    with open(filepath, "rb") as f:
        audio_data = f.read()
    return Response(audio_data, mimetype="audio/mpeg")


@app.route("/test", methods=["GET"])
def test():
    """Endpoint de test pour vérifier que le serveur tourne."""
    reponse = appeler_llm("Comment vérifier mon solde Orange ?")
    return json.dumps({
        "status": "ok",
        "telma": reponse
    }, ensure_ascii=False)


@app.route("/", methods=["GET"])
def home():
    return json.dumps({"status": "TELMA API en ligne", "version": "1.0"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)


