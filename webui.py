import gradio as gr
import os
import torchaudio
import torch
from fireredtts.fireredtts import FireRedTTS
from datetime import datetime
import re

# 初始化 FireRedTTS
try:
    print("Initializing FireRedTTS...")
    tts = FireRedTTS(
        config_path="configs/config_24k.json",
        pretrained_path="pretrained_models",  # 替换为实际的预训练模型路径
    )
    print("FireRedTTS initialized successfully.")
except Exception as e:
    print(f"Error initializing FireRedTTS: {e}")
    raise e

# 创建输出和引导音频目录
output_dir = "./outputs"
reference_dir = "./reference_audios"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(reference_dir, exist_ok=True)

def get_reference_audios():
    """
    获取引导音频目录下的所有 wav 文件
    """
    if not os.path.exists(reference_dir):
        return []
    
    wav_files = [f for f in os.listdir(reference_dir) if f.lower().endswith('.wav')]
    if not wav_files:
        print("Warning: No wav files found in reference_audios directory")
        return []
    
    return [os.path.join(reference_dir, f) for f in wav_files]

def update_audio_input(choice):
    """
    根据下拉菜单选择更新音频输入
    """
    if not choice:
        return None
    if choice == "请选择引导音频...":
        return None
    return choice

def clean_text(text):
    """
    清理文本，去除重复的标点符号
    """
    # 定义所有标点符号
    punctuations = ['。', '！', '？', '!', '?', '.', '；', ';', '，', ',']
    
    # 去除连续的标点符号，只保留第一个
    cleaned_text = text
    for punct in punctuations:
        while punct + punct in cleaned_text:
            cleaned_text = cleaned_text.replace(punct + punct, punct)
    
    # 去除不同标点的重复
    for i, punct1 in enumerate(punctuations):
        for punct2 in punctuations[i+1:]:
            if punct1 + punct2 in cleaned_text:
                cleaned_text = cleaned_text.replace(punct1 + punct2, punct1)
            if punct2 + punct1 in cleaned_text:
                cleaned_text = cleaned_text.replace(punct2 + punct1, punct1)
    
    return cleaned_text.strip()

def split_text(text, base_length=30):
    """
    将长文本按照自然段落切分，优化后的版本
    """
    # 定义分隔符优先级
    primary_separators = ['。', '！', '？', '!', '?', '.']    # 主要分隔符
    secondary_separators = ['；', ';']                        # 次要分隔符
    tertiary_separators = ['，', ',']                        # 第三级分隔符
    
    segments = []
    remaining_text = clean_text(text.strip())
    
    # 按主要分隔符分割成句子
    while remaining_text:
        # 寻找最近的主要分隔符
        found_pos = -1
        found_sep = None
        
        for sep in primary_separators:
            pos = remaining_text.find(sep)
            if pos != -1 and (found_pos == -1 or pos < found_pos):
                found_pos = pos
                found_sep = sep
        
        if found_sep:
            # 找到分隔符，添加这个句子
            sentence = clean_text(remaining_text[:found_pos + len(found_sep)])
            if sentence:  # 只添加非空句子
                segments.append(sentence)
            remaining_text = remaining_text[found_pos + len(found_sep):].strip()
        else:
            # 处理剩余文本
            if len(remaining_text) > base_length:
                # 尝试用次要分隔符分割
                sub_segments = split_by_secondary_separators(
                    remaining_text, 
                    secondary_separators + tertiary_separators, 
                    base_length
                )
                segments.extend(sub_segments)
            else:
                segments.append(clean_text(remaining_text))
            break
    
    # 合并过短的段落
    final_segments = []
    current = ''
    
    for seg in segments:
        if not current:
            current = seg
        elif len(current) + len(seg) <= base_length:
            current += seg
        else:
            final_segments.append(clean_text(current))
            current = seg
    
    if current:
        final_segments.append(clean_text(current))
    
    # 确保没有重复的段落
    return list(dict.fromkeys(final_segments))

def split_by_secondary_separators(text, separators, base_length):
    """
    使用次要分隔符进行分割
    """
    segments = []
    current_segment = ''
    
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            for i, part in enumerate(parts):
                if not part.strip():
                    continue
                    
                part = clean_text(part + (sep if i < len(parts)-1 else ''))
                
                if len(current_segment + part) <= base_length:
                    current_segment += part
                else:
                    if current_segment:
                        segments.append(clean_text(current_segment))
                    current_segment = part
            
            if current_segment:
                segments.append(clean_text(current_segment))
            return segments
    
    # 如果没有找到分隔符，强制按长度分割
    while text:
        segments.append(clean_text(text[:base_length]))
        text = text[base_length:].strip()
    
    return segments

def merge_audio(audio_segments):
    """
    合并多个音频段落
    Args:
        audio_segments: 音频张量列表
    Returns:
        merged_audio: 合并后的音频张量
    """
    return torch.cat(audio_segments, dim=1)

# 定义生成 TTS 音频的函数
def generate_tts(prompt_wav_path, text, lang, segment_length=40):
    # 从显示文本中提取实际的语言代码
    lang_code = lang.split()[0]  # 获取第一个空格前的内容（即 "zh"、"en" 或 "auto"）
    
    try:
        print(f"Input prompt_wav_path: {prompt_wav_path}")
        print(f"Input text: {text}")
        print(f"Input lang: {lang_code}")  # 使用提取的语言代码
        print(f"Segment length: {segment_length}")
        
        # 使用传入的分段长度参数
        segments = split_text(text, base_length=segment_length)
        print(f"\nText split into {len(segments)} segments:")
        for i, seg in enumerate(segments, 1):
            print(f"Segment {i}: [{len(seg)}字] {seg}")
        
        # 存储每段音频
        audio_segments = []
        
        # 处理每个文本段落
        for i, segment in enumerate(segments):
            print(f"Processing segment {i+1}/{len(segments)}: {segment}")
            
            rec_wavs = tts.synthesize(
                prompt_wav=prompt_wav_path,
                text=segment,
                lang=lang_code,
            )
            audio_segments.append(rec_wavs.detach().cpu())
        
        # 合并所有音频段落
        if len(audio_segments) > 1:
            print("Merging audio segments...")
            rec_wavs = merge_audio(audio_segments)
        else:
            rec_wavs = audio_segments[0]
        
        # 使用时间戳命名文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_wav_path = os.path.join(output_dir, f"output_{timestamp}.wav")
        
        print(f"Saving output audio to {out_wav_path}...")
        torchaudio.save(
            out_wav_path, 
            rec_wavs, 
            sample_rate=24000,
            encoding='PCM_S', 
            bits_per_sample=16
        )
        print("Audio saved successfully.")
        
        # 返回绝对路径
        return os.path.abspath(out_wav_path)
    except Exception as e:
        error_message = f"Error during TTS synthesis: {str(e)}"
        print(error_message)
        return None

def get_language_choices():
    """
    获取支持的语言选项，返回选项列表和默认值
    """
    return [
        {"value": "zh", "label": "zh (中文)"},
        {"value": "en", "label": "en (英文)"},
        {"value": "auto", "label": "auto (自动检测)"}
    ]

# 定义自定义主题
custom_theme = gr.themes.Soft(
    primary_hue="indigo",          # 主色调：靛蓝色
    secondary_hue="slate",         # 次要色调：石板灰
    neutral_hue="slate",           # 中性色调：石板灰
    font=("Helvetica", "Microsoft YaHei", "sans-serif")  # 字体设置
)

# 创建 Gradio 界面
with gr.Blocks(theme=custom_theme, css="""
    :root {
        --primary-50: #eef2ff;
        --primary-100: #e0e7ff;
        --primary-200: #c7d2fe;
        --primary-300: #a5b4fc;
        --primary-400: #818cf8;
        --primary-500: #6366f1;
        --primary-600: #4f46e5;
        --primary-700: #4338ca;
        --neutral-50: #f8fafc;
        --neutral-100: #f1f5f9;
        --neutral-200: #e2e8f0;
        --neutral-300: #cbd5e1;
        --neutral-400: #94a3b8;
        --neutral-500: #64748b;
        --neutral-600: #475569;
        --neutral-700: #334155;
        --neutral-800: #1e293b;
    }
    
    .gradio-container {
        max-width: 1200px !important;
        margin: auto;
        padding: 1rem;
    }
    
    .main-header {
        text-align: center;
        margin-bottom: 0.2rem;
        color: var(--primary-600);
        font-size: 1.8rem;
        font-weight: bold;
        padding: 0.3rem;
    }
    
    .description {
        text-align: center;
        color: var(--neutral-600);
        margin-bottom: 0.8rem;
        font-size: 0.9rem;
        padding: 0;
        margin-top: -0.2rem;
    }
    
    .header-container {
        margin-bottom: 1rem;
        background: var(--neutral-50);
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    .gr-button-primary {
        background: var(--primary-500) !important;
        border: none !important;
        color: white !important;
        padding: 0.75rem 1.5rem !important;
        font-size: 1.1rem !important;
    }
    
    .gr-button-primary:hover {
        background: var(--primary-600) !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .gr-input, .gr-textarea {
        border: 1px solid var(--neutral-200) !important;
        border-radius: 8px !important;
        background: var(--neutral-50) !important;
    }
    
    .gr-input:focus, .gr-textarea:focus {
        border-color: var(--primary-300) !important;
        box-shadow: 0 0 0 3px var(--primary-100) !important;
    }
    
    .gr-slider {
        margin-top: 1rem !important;
        margin-bottom: 2rem !important;
    }
    
    .gr-form {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        gap: 0.5rem !important;
    }
    
    .gr-box {
        border-radius: 8px;
        background: var(--neutral-50);
    }
    
    .reference-note {
        color: var(--neutral-600);
        font-size: 0.9rem;
        margin-top: 0.5rem;
        font-style: italic;
    }
    
    .gr-row {
        margin-bottom: 0.5rem !important;
    }
    
    .gr-input-label, .gr-dropdown-label {
        margin-bottom: 0.2rem !important;
    }
""") as app:
    with gr.Row(elem_classes="header-container"):
        with gr.Column():
            gr.Markdown(
                "# FireRedTTS 语音合成系统", 
                elem_classes=["main-header"]
            )
            gr.Markdown(
                "基于深度学习的高质量语音合成系统，支持中文和英文。",
                elem_classes=["description"]
            )
    
    with gr.Row():
        with gr.Column(scale=1):
            # 添加引导音频选择下拉菜单
            reference_files = get_reference_audios()
            choices = ["请选择引导音频..."] + reference_files if reference_files else ["请选择引导音频..."]
            
            reference_dropdown = gr.Dropdown(
                choices=choices,
                value="请选择引导音频..." if choices else None,
                label="选择引导音频",
                info="从 reference_audios 目录中选择预设的引导音频"
            )
            
            input_prompt_wav = gr.Audio(
                label="或上传新的引导音频 (wav 格式)", 
                type="filepath",
                elem_classes="audio-input"
            )
            
            if not reference_files:
                gr.Markdown(
                    "提示：您可以将常用的引导音频文件放在 reference_audios 目录下",
                    elem_classes=["reference-note"]
                )
        
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label="输入文本", 
                placeholder="请输入需要合成的文本...",
                lines=4
            )
            input_lang = gr.Dropdown(
                choices=[choice["label"] for choice in get_language_choices()],
                value="zh (中文)",
                label="选择语言",
                info="选择文本语言或使用自动检测"
            )
    
    with gr.Row():
        segment_length = gr.Slider(
            minimum=20,
            maximum=200,
            value=40,
            step=10,
            label="分段长度（字符数）",
            info="建议设置在 40-100 之间，数值越大段落越长"
        )
    
    with gr.Row():
        output_audio = gr.Audio(
            label="生成的音频",
            type="filepath",
            autoplay=True
        )
    
    with gr.Row():
        submit_button = gr.Button(
            "生成语音",
            variant="primary",
            size="lg"
        )
    
    # 添加下拉菜单选择事件
    reference_dropdown.change(
        fn=update_audio_input,
        inputs=[reference_dropdown],
        outputs=[input_prompt_wav]
    )
    
    # 生成按钮事件
    submit_button.click(
        fn=generate_tts,
        inputs=[
            input_prompt_wav,  # 如果下拉菜单选择了音频，这里会被更新
            input_text,
            input_lang,
            segment_length
        ],
        outputs=output_audio
    )

# 运行 Gradio 应用
print("Launching Gradio app...")
app.launch(
    server_name="127.0.0.1",
    server_port=6006,
    inbrowser=True
)
print("Gradio app running on port 6006...")
