import gradio as gr
import os
from dotenv import load_dotenv
from brain_of_doctor_ai import analyze_with_vision, find_nearby_doctors, suggest_medicines, calculate_tb_risk
from voice_of_the_patient import transcribe_with_grog
from voice_of_the_doctor import text_to_speech_with_elevenlabs
from report_generator import create_prescription_pdf

load_dotenv()

# --- 🌑 DARK MODE THEME (Clean & Fixed) ---
dark_mode_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-page: #09090b;       
    --bg-card: #18181b;       
    --bg-input: #27272a;      
    --border-main: #3f3f46;   
    --text-primary: #fafafa;  
    --text-secondary: #a1a1aa;
    --accent: #3b82f6;        
    --radius: 8px;
}

* { font-family: 'Inter', sans-serif !important; }

body, .gradio-container {
    background-color: var(--bg-page) !important;
    color: var(--text-primary) !important;
}

/* CARDS - Removed 'height: 100%' to fix the giant empty box issue */
.dom-card {
    background: var(--bg-card);
    border: 1px solid var(--border-main);
    border-radius: var(--radius);
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 15px; /* Adds nice spacing between elements inside card */
}

/* TYPOGRAPHY */
.h1 { font-size: 1.25rem; font-weight: 700; color: var(--text-primary); margin-bottom: 6px; }
.p { font-size: 0.875rem; color: var(--text-secondary); margin-bottom: 20px; }
.label { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #71717a; margin-bottom: 8px; display: block; }

/* FIXING INPUTS & BOXES */
.gr-box, .gr-input, textarea, input, .gr-form {
    background-color: var(--bg-input) !important;
    border: 1px solid var(--border-main) !important;
    color: var(--text-primary) !important;
}
/* Fix for file download box looking weird */
.gr-file {
    background-color: var(--bg-input) !important;
    border-color: var(--border-main) !important;
}

/* BUTTONS */
button.primary {
    background-color: #2563eb !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
}
button.primary:hover { background-color: #1d4ed8 !important; }

/* TABS */
.tabs { background: transparent; border-bottom: 1px solid var(--border-main); margin-bottom: 30px; }
.tab-nav button { color: var(--text-secondary) !important; font-weight: 500; }
.tab-nav button.selected { color: #fff !important; border-bottom: 2px solid #3b82f6 !important; background: transparent !important; }

/* WAITING STATE - Simple & Clean */
.waiting-text {
    color: #52525b;
    font-style: italic;
    font-size: 0.9rem;
    padding: 20px;
    text-align: center;
    border: 1px dashed #3f3f46;
    border-radius: 8px;
}
"""

# --- LOGIC FUNCTIONS ---
def format_risk_html(result):
    color = "#fff"; bg = "#18181b"; border = "#3f3f46"
    if "HIGH" in result['status']: color = "#f87171"; bg = "#450a0a"; border = "#7f1d1d" 
    elif "MODERATE" in result['status']: color = "#fbbf24"; bg = "#451a03"; border = "#78350f"
    else: color = "#34d399"; bg = "#064e3b"; border = "#065f46"

    return f"""
    <div style="border: 1px solid {border}; background: {bg}; padding: 24px; border-radius: 8px;">
        <div style="font-family:'Inter'; font-size:0.7rem; font-weight:700; text-transform:uppercase; color:{color}; opacity:0.9; margin-bottom:6px;">Diagnostic Result</div>
        <div style="font-family:'Inter'; font-size:1.75rem; font-weight:700; color:#fff;">{result['status']}</div>
        <div style="margin-top:16px; border-top:1px solid {border}; padding-top:12px; font-family:'JetBrains Mono'; font-size:0.75rem; color:#a1a1aa;">
            CONFIDENCE: {result['total_score']} <br> {result['breakdown']}
        </div>
    </div>
    """

def logic_tb_scan_pretty(xray_image, cough_audio, language):
    if not xray_image and not cough_audio:
        return "<div style='color:#f87171; text-align:center;'>⚠️ Input Required</div>", "N/A", None
    
    transcript = ""
    if cough_audio:
        transcript = transcribe_with_grog(os.getenv("GROQ_API_KEY"), cough_audio, "whisper-large-v3")
    
    result = calculate_tb_risk(cough_audio, xray_image, transcript)
    
    from brain_of_doctor_ai import GROQ_API_KEY, Groq
    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        messages=[{"role": "user", "content": f"Summarize TB risk ({result['status']}) for {language}. Strict clinical tone. Max 3 sentences."}],
        model="llama-3.3-70b-versatile"
    )
    return format_risk_html(result), resp.choices[0].message.content.replace("**", ""), find_nearby_doctors("pulmonologist")

def logic_general(audio, image, language):
    transcript = "No audio."
    if audio: transcript = transcribe_with_grog(os.getenv("GROQ_API_KEY"), audio, "whisper-large-v3")
    if image:
        resp = analyze_with_vision(image, "general", user_text=transcript, language=language)
    else:
        from brain_of_doctor_ai import GROQ_API_KEY, Groq
        client = Groq(api_key=GROQ_API_KEY)
        r = client.chat.completions.create(
            messages=[{"role": "user", "content": f"Patient: {transcript}. Reply in {language}."}],
            model="llama-3.3-70b-versatile"
        )
        resp = r.choices[0].message.content.replace("**", "")
    audio_path = "response.mp3"
    text_to_speech_with_elevenlabs(resp[:450], audio_path)
    return transcript, resp, audio_path

def logic_radiology(image, prompt_text, language):
    if not image: return "Please upload a Scan.", None
    analysis = analyze_with_vision(image, "radiology", user_text=prompt_text, language=language)
    return analysis, find_nearby_doctors("radiologist")

def logic_skin(image, language):
    if not image: return "Upload image.", None
    return analyze_with_vision(image, "skin", language=language), find_nearby_doctors("dermatologist")

def logic_blood(image, language):
    if not image: return "Upload report.", None
    return analyze_with_vision(image, "blood", language=language), find_nearby_doctors("hematologist")

def logic_pharmacy(text, audio, language):
    if audio: text = transcribe_with_grog(os.getenv("GROQ_API_KEY"), audio, "whisper-large-v3")
    meds = suggest_medicines(text, language)
    return text, meds, create_prescription_pdf(text, meds)


# --- UI BUILDER ---
with gr.Blocks(theme=gr.themes.Base(), css=dark_mode_css, title="MedGaurd.AI") as demo:
    
    with gr.Row(elem_classes="tabs"):
        with gr.Column(scale=1):
            gr.HTML("""
            <div style="display: flex; align-items: center; gap: 12px; padding: 20px 0;">
                <div style="width: 32px; height: 32px; background: #2563eb; border-radius: 8px; display:flex; align-items:center; justify-content:center;">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5"><path d="M12 2v20M2 12h20"/></svg>
                </div>
                <div style="font-weight: 600; font-size: 1.1rem; color: #fff;">MedGaurd.AI</div>
                <div style="font-size: 0.6rem; font-weight: 700; background: #27272a; padding: 3px 8px; border-radius: 99px; color: #a1a1aa; border: 1px solid #3f3f46;">PRO</div>
            </div>
            """)
        with gr.Column(scale=2): pass
        with gr.Column(scale=1):
             lang_dropdown = gr.Dropdown(["English", "Hindi"], value="English", show_label=False, interactive=True, container=False)

    with gr.Tabs():
        
        # TAB 1: GENERAL
        with gr.TabItem("General"):
            with gr.Row():
                with gr.Column(elem_classes="dom-card"):
                    gr.HTML("<div class='label'>INPUT</div><div class='h1'>Symptom Analysis</div>")
                    in_aud = gr.Audio(sources=["microphone"], type="filepath", show_label=False)
                    in_cam = gr.Image(type="filepath", show_label=False)
                    btn_gen = gr.Button("Start Session", variant="primary")
                with gr.Column(elem_classes="dom-card"):
                    gr.HTML("<div class='label'>OUTPUT</div><div class='h1'>Assessment</div>")
                    out_gen = gr.HTML("<div class='waiting-text'>Waiting...</div>")
                    out_aud = gr.Audio(label="Audio Output", autoplay=True, show_label=False)
                    out_trans = gr.Textbox(label="Transcript", visible=True)
            btn_gen.click(logic_general, [in_aud, in_cam, lang_dropdown], [out_trans, out_gen, out_aud])

        # TAB 2: RADIOLOGY
        with gr.TabItem("Radiology"):
            with gr.Row():
                with gr.Column(elem_classes="dom-card"):
                    gr.HTML("<div class='label'>SCAN UPLOAD</div><div class='h1'>Medical Imaging</div>")
                    rad_in = gr.Image(type="filepath", show_label=False)
                    rad_text = gr.Textbox(placeholder="Clinical notes...", show_label=False)
                    btn_rad = gr.Button("Generate Report", variant="primary")
                with gr.Column(elem_classes="dom-card"):
                    gr.HTML("<div class='label'>FINDINGS</div>")
                    rad_out = gr.Markdown(value="Waiting...")
                    rad_map = gr.Markdown()
            btn_rad.click(logic_radiology, [rad_in, rad_text, lang_dropdown], [rad_out, rad_map])

        # TAB 3: PHARMACY
        with gr.TabItem("Pharmacy"):
            with gr.Row():
                with gr.Column(elem_classes="dom-card"):
                    gr.HTML("<div class='label'>INPUT</div><div class='h1'>Prescription Engine</div>")
                    in_mic = gr.Audio(sources=["microphone"], type="filepath", show_label=False)
                    in_sym = gr.Textbox(placeholder="Describe symptoms...", show_label=False)
                    btn_rx = gr.Button("Search Protocol", variant="primary")
                with gr.Column(elem_classes="dom-card"):
                    gr.HTML("<div class='label'>CARE PLAN</div>")
                    out_rx = gr.Markdown("Waiting...")
                    out_pdf = gr.File(label="Download PDF")
            btn_rx.click(logic_pharmacy, [in_sym, in_mic, lang_dropdown], [in_sym, out_rx, out_pdf])

        # TAB 4: DERMATOLOGY
        with gr.TabItem("Dermatology"):
            with gr.Row():
                with gr.Column(elem_classes="dom-card"):
                    gr.HTML("<div class='label'>SCAN</div><div class='h1'>Lesion Analysis</div>")
                    in_skin = gr.Image(type="filepath", show_label=False)
                    btn_skin = gr.Button("Analyze", variant="primary")
                with gr.Column(elem_classes="dom-card"):
                    gr.HTML("<div class='label'>REPORT</div>")
                    out_skin = gr.Markdown(value="Waiting...")
                    out_map_s = gr.Markdown() 
            btn_skin.click(logic_skin, [in_skin, lang_dropdown], [out_skin, out_map_s])

        # TAB 5: PATHOLOGY
        with gr.TabItem("Pathology"):
            with gr.Row():
                with gr.Column(elem_classes="dom-card"):
                    gr.HTML("<div class='label'>UPLOAD</div><div class='h1'>Lab Reports</div>")
                    in_bl = gr.Image(type="filepath", show_label=False)
                    btn_bl = gr.Button("Analyze", variant="primary")
                with gr.Column(elem_classes="dom-card"):
                    gr.HTML("<div class='label'>RESULTS</div>")
                    out_bl = gr.HTML("<div class='waiting-text'>Waiting...</div>")
                    out_map_b = gr.Markdown()
            btn_bl.click(logic_blood, [in_bl, lang_dropdown], [out_bl, out_map_b])

        # TAB 6: TB SCANNER (MOVED TO LAST POSITION)
        with gr.TabItem("TB Scanner"):
            with gr.Row():
                with gr.Column(elem_classes="dom-card"): 
                    gr.HTML("<div class='label'>01 INPUT DATA</div>")
                    gr.HTML("<div class='h1'>Patient Diagnostics</div>")
                    gr.HTML("<div class='p'>Upload chest imaging (PA-View) and record respiratory audio for triangulation.</div>")
                    tb_xray = gr.Image(type="filepath", label="Chest X-Ray", height=220, show_label=False)
                    tb_audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio Sample", show_label=False)
                    btn_tb = gr.Button("Deploy Analysis Model", variant="primary")

                with gr.Column(elem_classes="dom-card"):
                    gr.HTML("<div class='label'>02 AI RESULTS</div>")
                    gr.HTML("<div class='h1'>Clinical Report</div>")
                    gr.HTML("<div class='p'>Real-time inference • Model v2.4 (Dark)</div>")
                    
                    # REVERTED TO WAITING TEXT
                    tb_stats = gr.HTML(value="<div class='waiting-text'>Waiting...</div>")
                    
                    gr.HTML("<div style='height:24px'></div>")
                    gr.HTML("<div class='label'>INTERPRETATION</div>")
                    tb_expl = gr.Markdown("Analysis pending...")
                    
                    gr.HTML("<div class='label'>REFERRAL</div>")
                    tb_map = gr.Markdown("No referral generated.")
            
            btn_tb.click(logic_tb_scan_pretty, [tb_xray, tb_audio, lang_dropdown], [tb_stats, tb_expl, tb_map])

    gr.HTML("<div style='text-align:center; padding:40px 0; border-top:1px solid #3f3f46; margin-top:40px; color:#52525b; font-size:11px;'>SECURED BY DR. AI CLOUD</div>")

if __name__ == "__main__":
    demo.launch()