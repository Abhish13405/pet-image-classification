"""
Automated PowerPoint Presentation Generator
Generates a modern, widescreen (16:9) professional presentation deck
for Abhishek Kushwaha's Pet Image Classification project.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# Paths
BASE_DIR = r"C:\Users\Abhi\Desktop\internship\pet-image-classification"
OUTPUT_PPTX = os.path.join(BASE_DIR, "Pet_Image_Classification_Abhishek_Kushwaha.pptx")
HISTORY_PLOT = os.path.join(BASE_DIR, "model", "training_history.png")
SAMPLE_CAT = os.path.join(BASE_DIR, "cat", "0.jpg")
SAMPLE_DOG = os.path.join(BASE_DIR, "dog", "0.jpg")

# Professional Color Palette
COLOR_DARK_BG = RGBColor(15, 23, 42)       # Slate 900
COLOR_LIGHT_BG = RGBColor(248, 250, 252)   # Slate 50
COLOR_PRIMARY = RGBColor(37, 99, 235)      # Blue 600
COLOR_PRIMARY_LIGHT = RGBColor(219, 234, 254) # Blue 100
COLOR_ACCENT = RGBColor(14, 165, 233)      # Sky 500
COLOR_TEXT_DARK = RGBColor(30, 41, 59)     # Slate 800
COLOR_TEXT_MUTED = RGBColor(100, 116, 139) # Slate 500
COLOR_WHITE = RGBColor(255, 255, 255)
COLOR_CARD_BG = RGBColor(255, 255, 255)
COLOR_CARD_BORDER = RGBColor(226, 232, 240) # Slate 200
COLOR_GREEN = RGBColor(16, 185, 129)       # Emerald 500


def create_presentation():
    prs = Presentation()
    # 16:9 Widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def add_header(slide, title_text, category_text="PET IMAGE CLASSIFICATION • MACHINE LEARNING"):
        # Category tag
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.5), Inches(0.35))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        tf_cat.margin_left = tf_cat.margin_top = tf_cat.margin_right = tf_cat.margin_bottom = 0
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.name = "Arial"
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = COLOR_PRIMARY

        # Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.75), Inches(11.5), Inches(0.7))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        tf_title.margin_left = tf_title.margin_top = tf_title.margin_right = tf_title.margin_bottom = 0
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.name = "Arial"
        p_title.font.size = Pt(26)
        p_title.font.bold = True
        p_title.font.color.rgb = COLOR_TEXT_DARK

    def set_slide_background(slide, color):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = color
        bg.line.color.rgb = color
        return bg

    def add_card(slide, left, top, width, height, bg_color=COLOR_CARD_BG, border_color=COLOR_CARD_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        if border_color:
            card.line.color.rgb = border_color
            card.line.width = Pt(1.5)
        else:
            card.line.fill.background()
        return card

    # =========================================================================
    # SLIDE 1: Title Slide (Dark Theme)
    # =========================================================================
    slide1 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide1, COLOR_DARK_BG)

    # Accent decorative bar
    bar = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.2), Inches(1.5), Inches(0.8), Inches(0.1))
    bar.fill.solid()
    bar.fill.fore_color.rgb = COLOR_ACCENT
    bar.line.fill.background()

    # Main Title
    t_box = slide1.shapes.add_textbox(Inches(1.2), Inches(1.8), Inches(11.0), Inches(2.2))
    tf = t_box.text_frame
    tf.word_wrap = True
    p1 = tf.paragraphs[0]
    p1.text = "Pet Image Classification System"
    p1.font.name = "Arial"
    p1.font.size = Pt(44)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_WHITE

    p2 = tf.add_paragraph()
    p2.text = "Deep Transfer Learning with MobileNetV2 & Interactive Streamlit Deployment"
    p2.font.name = "Arial"
    p2.font.size = Pt(20)
    p2.font.color.rgb = COLOR_ACCENT
    p2.space_before = Pt(14)

    # Presenter Card
    card1 = add_card(slide1, Inches(1.2), Inches(4.5), Inches(10.9), Inches(2.0),
                     bg_color=RGBColor(30, 41, 59), border_color=RGBColor(51, 65, 85))
    
    info_box = slide1.shapes.add_textbox(Inches(1.5), Inches(4.7), Inches(10.3), Inches(1.6))
    tf_info = info_box.text_frame
    tf_info.word_wrap = True
    
    p = tf_info.paragraphs[0]
    p.text = "PRESENTER DETAILS"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT
    
    p = tf_info.add_paragraph()
    p.text = "Abhishek Kushwaha"
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = COLOR_WHITE
    p.space_before = Pt(6)

    p = tf_info.add_paragraph()
    p.text = "B.Tech CSE (Cloud Computing & Machine Learning - CCML) • Section 4-A"
    p.font.size = Pt(14)
    p.font.color.rgb = RGBColor(203, 213, 225)
    p.space_before = Pt(4)

    # =========================================================================
    # SLIDE 2: Executive Summary / Project Overview
    # =========================================================================
    slide2 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide2, COLOR_LIGHT_BG)
    add_header(slide2, "Project Overview & Executive Summary")

    cards_data = [
        ("01", "The Objective", 
         "Build an end-to-end Computer Vision system capable of classifying pet photos (Cat vs Dog) with near-zero latency, production-ready accuracy, and zero infrastructure cost.",
         COLOR_PRIMARY),
        ("02", "The Approach", 
         "Leverage Deep Transfer Learning using Google's MobileNetV2 backbone, paired with dynamic runtime Data Augmentation and a custom two-phase fine-tuning strategy.",
         COLOR_ACCENT),
        ("03", "The Delivery", 
         "Deploy an interactive web application built with Streamlit, hosted on Streamlit Cloud with public accessibility, CI/CD with GitHub, and a standalone CLI tool.",
         COLOR_GREEN)
    ]

    col_width = Inches(3.75)
    gap = Inches(0.3)
    top_pos = Inches(1.8)
    card_h = Inches(4.8)

    for i, (num, heading, desc, col) in enumerate(cards_data):
        left_pos = Inches(0.8) + i * (col_width + gap)
        add_card(slide2, left_pos, top_pos, col_width, card_h)

        tb = slide2.shapes.add_textbox(left_pos + Inches(0.3), top_pos + Inches(0.4), col_width - Inches(0.6), card_h - Inches(0.8))
        tf = tb.text_frame
        tf.word_wrap = True

        p = tf.paragraphs[0]
        p.text = num
        p.font.size = Pt(36)
        p.font.bold = True
        p.font.color.rgb = col

        p = tf.add_paragraph()
        p.text = heading
        p.font.size = Pt(20)
        p.font.bold = True
        p.font.color.rgb = COLOR_TEXT_DARK
        p.space_before = Pt(14)

        p = tf.add_paragraph()
        p.text = desc
        p.font.size = Pt(14)
        p.font.color.rgb = COLOR_TEXT_MUTED
        p.space_before = Pt(12)

    # =========================================================================
    # SLIDE 3: Problem Statement & Small-Data Challenge
    # =========================================================================
    slide3 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide3, COLOR_LIGHT_BG)
    add_header(slide3, "Problem Statement: The Small-Data Challenge")

    # Left Card: Traditional CNN Dilemma
    add_card(slide3, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8))
    tb_left = slide3.shapes.add_textbox(Inches(1.1), Inches(2.1), Inches(5.0), Inches(4.2))
    tf_l = tb_left.text_frame
    tf_l.word_wrap = True
    
    p = tf_l.paragraphs[0]
    p.text = "Traditional Deep Learning Challenges"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = RGBColor(220, 38, 38)

    bullets_left = [
        "Massive Data Appetite: Scratch CNNs require 10,000+ labeled images to converge effectively.",
        "Catastrophic Overfitting: When data is small, standard networks memorize pixels instead of learning visual concepts.",
        "Heavy Compute Cost: Training deep architectures requires high-end GPUs, hours of compute, and heavy electricity.",
        "Model Bloat: Enterprise architectures (VGG, ResNet) exceed 500 MB, making mobile edge deployment impractical."
    ]
    for b in bullets_left:
        p = tf_l.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(13)
        p.font.color.rgb = COLOR_TEXT_MUTED
        p.space_before = Pt(10)

    # Right Card: Our Transfer Learning Solution
    add_card(slide3, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8))
    tb_right = slide3.shapes.add_textbox(Inches(7.1), Inches(2.1), Inches(5.1), Inches(4.2))
    tf_r = tb_right.text_frame
    tf_r.word_wrap = True

    p = tf_r.paragraphs[0]
    p.text = "Our Transfer Learning Solution"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    bullets_right = [
        "Pre-Trained Intelligence: MobileNetV2 already understands 1.4 million ImageNet images, pre-learning edges, shapes, and textures.",
        "Few-Shot Mastery: Achieves 100% accuracy on small datasets (42 images) through inductive knowledge transfer.",
        "Lightweight Footprint: Only 20.5 MB file size with 2.26 million parameters, running at sub-80ms CPU speeds.",
        "Production-Grade Stability: Incorporates Data Augmentation and Dropout to guarantee robust generalization."
    ]
    for b in bullets_right:
        p = tf_r.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(13)
        p.font.color.rgb = COLOR_TEXT_MUTED
        p.space_before = Pt(10)

    # =========================================================================
    # SLIDE 4: End-to-End System Architecture
    # =========================================================================
    slide4 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide4, COLOR_LIGHT_BG)
    add_header(slide4, "End-to-End System Architecture & Pipeline")

    steps = [
        ("1. Data Pipeline", "• Balanced 80/20 train/val split\n• 160x160 resolution\n• Stratified batching", COLOR_PRIMARY),
        ("2. Augmentation", "• Random horizontal flips\n• 15% random rotation\n• 10% dynamic zoom", COLOR_ACCENT),
        ("3. MobileNetV2 Base", "• Frozen ImageNet weights\n• Inverted residual blocks\n• High-level feature map", RGBColor(147, 51, 234)),
        ("4. Custom Head", "• GlobalAvgPool2D\n• Dropout (0.25)\n• Dense(1, Sigmoid)", RGBColor(217, 119, 6)),
        ("5. Deployment", "• Saved .keras model\n• Streamlit Cloud Web UI\n• CLI prediction tool", COLOR_GREEN)
    ]

    step_w = Inches(2.25)
    step_gap = Inches(0.15)
    for i, (title, details, col) in enumerate(steps):
        pos_l = Inches(0.7) + i * (step_w + step_gap)
        add_card(slide4, pos_l, Inches(2.0), step_w, Inches(4.5))

        tb = slide4.shapes.add_textbox(pos_l + Inches(0.15), Inches(2.2), step_w - Inches(0.3), Inches(4.1))
        tf = tb.text_frame
        tf.word_wrap = True

        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = col

        p = tf.add_paragraph()
        p.text = details
        p.font.size = Pt(13)
        p.font.color.rgb = COLOR_TEXT_DARK
        p.space_before = Pt(14)

    # =========================================================================
    # SLIDE 5: Why MobileNetV2?
    # =========================================================================
    slide5 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide5, COLOR_LIGHT_BG)
    add_header(slide5, "Why MobileNetV2? Architectural Advantage")

    # Table Comparison Card
    add_card(slide5, Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.8))
    
    tb = slide5.shapes.add_textbox(Inches(1.1), Inches(2.1), Inches(11.1), Inches(4.2))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "Key Architectural Innovations in MobileNetV2:"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    bullets = [
        "Depthwise Separable Convolutions: Splits conventional convolution into depthwise and pointwise stages, slashing computation by 8x to 9x.",
        "Inverted Residual Blocks: Connects bottlenecks directly, enabling efficient gradient propagation and preserving information across thin layers.",
        "Linear Bottlenecks: Prevents non-linearities (like ReLU) from destroying manifold representations in low-dimensional spaces.",
        "Edge-Native Deployment: Engineered specifically for smartphones, autonomous drones, and embedded edge systems with ultra-low power consumption."
    ]
    for b in bullets:
        p = tf.add_paragraph()
        p.text = "✔  " + b
        p.font.size = Pt(14)
        p.font.color.rgb = COLOR_TEXT_DARK
        p.space_before = Pt(12)

    # =========================================================================
    # SLIDE 6: Two-Phase Training Strategy
    # =========================================================================
    slide6 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide6, COLOR_LIGHT_BG)
    add_header(slide6, "Two-Phase Training & Fine-Tuning Strategy")

    # Phase 1 Card
    add_card(slide6, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8))
    tb_p1 = slide6.shapes.add_textbox(Inches(1.1), Inches(2.1), Inches(5.0), Inches(4.2))
    tf1 = tb_p1.text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.text = "Phase 1: Feature Extraction"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    p1_items = [
        "Base Model Frozen: All 154 layers of MobileNetV2 set to trainable=False.",
        "Focus: Train only randomly initialized top classification head.",
        "Optimizer: Adam with learning rate = 0.001.",
        "Duration: 15 Epochs.",
        "Outcome: Preserved low-level edge/texture filters while allowing the new classifier to find optimal initial decision boundaries."
    ]
    for it in p1_items:
        p = tf1.add_paragraph()
        p.text = "• " + it
        p.font.size = Pt(13)
        p.font.color.rgb = COLOR_TEXT_MUTED
        p.space_before = Pt(8)

    # Phase 2 Card
    add_card(slide6, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8))
    tb_p2 = slide6.shapes.add_textbox(Inches(7.1), Inches(2.1), Inches(5.1), Inches(4.2))
    tf2 = tb_p2.text_frame
    tf2.word_wrap = True

    p = tf2.paragraphs[0]
    p.text = "Phase 2: High-Level Fine-Tuning"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = COLOR_GREEN

    p2_items = [
        "Unfrozen Top 25 Layers: Higher-level convolutional blocks unfreezed for domain adaptation.",
        "Micro Learning Rate: Adam with LR = 1e-5 (prevents destroying pre-trained ImageNet weights).",
        "Duration: 10 Epochs (Epoch 16 to 25).",
        "Outcome: Adapts generic high-level animal features specifically to discriminating Cat vs Dog facial patterns.",
        "Final Validation Loss: Stabilized at a minimal 0.1049."
    ]
    for it in p2_items:
        p = tf2.add_paragraph()
        p.text = "• " + it
        p.font.size = Pt(13)
        p.font.color.rgb = COLOR_TEXT_MUTED
        p.space_before = Pt(8)

    # =========================================================================
    # SLIDE 7: Experimental Results & Training Metrics (With embedded chart)
    # =========================================================================
    slide7 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide7, COLOR_LIGHT_BG)
    add_header(slide7, "Experimental Results & Model Evaluation")

    # Left: Metric Cards
    add_card(slide7, Inches(0.8), Inches(1.8), Inches(4.2), Inches(4.8))
    tb_m = slide7.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(3.8), Inches(4.4))
    tf_m = tb_m.text_frame
    tf_m.word_wrap = True

    p = tf_m.paragraphs[0]
    p.text = "Key Metrics Achieved:"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    metrics = [
        ("Validation Accuracy", "100.00%", COLOR_GREEN),
        ("Validation Loss", "0.1049", COLOR_PRIMARY),
        ("Inference Latency", "< 80 ms", COLOR_TEXT_DARK),
        ("Model Size", "20.5 MB", COLOR_TEXT_DARK)
    ]
    for label, val, c in metrics:
        p = tf_m.add_paragraph()
        p.text = f"{label}: {val}"
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = c
        p.space_before = Pt(12)

    p = tf_m.add_paragraph()
    p.text = "\nSample Tests:\n• cat/0.jpg: 85.78% Cat\n• dog/0.jpg: 90.42% Dog"
    p.font.size = Pt(12)
    p.font.color.rgb = COLOR_TEXT_MUTED
    p.space_before = Pt(8)

    # Right: Embed training plot
    if os.path.exists(HISTORY_PLOT):
        add_card(slide7, Inches(5.3), Inches(1.8), Inches(7.2), Inches(4.8))
        slide7.shapes.add_picture(HISTORY_PLOT, Inches(5.5), Inches(2.0), width=Inches(6.8))

    # =========================================================================
    # SLIDE 8: Interactive Web Dashboard & Deployment
    # =========================================================================
    slide8 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide8, COLOR_LIGHT_BG)
    add_header(slide8, "Interactive Web Dashboard & Cloud Deployment")

    features = [
        ("Real-Time Drag & Drop", "Users can upload any JPG, JPEG, or PNG pet image directly through the web interface with instant visual feedback."),
        ("Catalog Test Gallery", "Built-in sample selector allows evaluating images from the dataset catalog with a single click without searching files."),
        ("Confidence Gauges", "Renders dynamic probability breakdown meters for both Cat and Dog, showing raw sigmoid activation confidence."),
        ("Streamlit Cloud Hosting", "Continuously deployed to Streamlit Community Cloud with automatic CI/CD sync from GitHub.")
    ]

    for i, (title, desc) in enumerate(features):
        pos_t = Inches(1.8) + (i % 2) * Inches(2.4)
        pos_l = Inches(0.8) + (i // 2) * Inches(5.9)
        add_card(slide8, pos_l, pos_t, Inches(5.6), Inches(2.1))

        tb = slide8.shapes.add_textbox(pos_l + Inches(0.3), pos_t + Inches(0.25), Inches(5.0), Inches(1.6))
        tf = tb.text_frame
        tf.word_wrap = True

        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(17)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY

        p = tf.add_paragraph()
        p.text = desc
        p.font.size = Pt(13)
        p.font.color.rgb = COLOR_TEXT_MUTED
        p.space_before = Pt(6)

    # =========================================================================
    # SLIDE 9: Industry Relevance & Scalability
    # =========================================================================
    slide9 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide9, COLOR_LIGHT_BG)
    add_header(slide9, "Industry Relevance: Data-Centric AI & Scalability")

    points = [
        ("Data-Centric AI In Action",
         "Rather than demanding millions of data points, modern AI engineering focuses on high-quality data pipelines, smart augmentations, and transfer learning."),
        ("Edge & IoT Deployment Ready",
         "The compact 20 MB footprint allows direct integration into smart cameras, automatic pet feeders, veterinary diagnostic kiosks, and mobile applications."),
        ("Fully Scalable tf.data Pipeline",
         "The ingestion pipeline is decoupled and dynamic. Feeding 10,000 new images requires zero architecture change; the model scales seamlessly.")
    ]

    for i, (title, desc) in enumerate(points):
        pos_t = Inches(1.8) + i * Inches(1.6)
        add_card(slide9, Inches(0.8), pos_t, Inches(11.7), Inches(1.4))

        tb = slide9.shapes.add_textbox(Inches(1.1), pos_t + Inches(0.2), Inches(11.1), Inches(1.0))
        tf = tb.text_frame
        tf.word_wrap = True

        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(17)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY

        p = tf.add_paragraph()
        p.text = desc
        p.font.size = Pt(13)
        p.font.color.rgb = COLOR_TEXT_MUTED
        p.space_before = Pt(4)

    # =========================================================================
    # SLIDE 10: Future Enhancements & Scope
    # =========================================================================
    slide10 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide10, COLOR_LIGHT_BG)
    add_header(slide10, "Future Enhancements & Technical Roadmap")

    future_cards = [
        ("Multi-Class Breed Recognition", "Expand classification to 120+ specific dog and cat breeds using the Stanford Dogs and Oxford-IIIT Pet datasets.", COLOR_PRIMARY),
        ("Object Detection (YOLO / SSD)", "Integrate bounding box localization to detect, count, and track multiple pets simultaneously in dynamic video streams.", COLOR_ACCENT),
        ("TFLite / ONNX Quantization", "Quantize model weights to INT8 to shrink model size below 5 MB for microcontrollers and native Android/iOS execution.", COLOR_GREEN),
        ("FastAPI Microservices", "Wrap the model in a high-concurrency asynchronous REST API for enterprise SaaS integration.", RGBColor(147, 51, 234))
    ]

    for i, (title, desc, col) in enumerate(future_cards):
        pos_t = Inches(1.8) + (i // 2) * Inches(2.4)
        pos_l = Inches(0.8) + (i % 2) * Inches(5.9)
        add_card(slide10, pos_l, pos_t, Inches(5.6), Inches(2.1))

        tb = slide10.shapes.add_textbox(pos_l + Inches(0.3), pos_t + Inches(0.25), Inches(5.0), Inches(1.6))
        tf = tb.text_frame
        tf.word_wrap = True

        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(17)
        p.font.bold = True
        p.font.color.rgb = col

        p = tf.add_paragraph()
        p.text = desc
        p.font.size = Pt(13)
        p.font.color.rgb = COLOR_TEXT_MUTED
        p.space_before = Pt(6)

    # =========================================================================
    # SLIDE 11: Conclusion
    # =========================================================================
    slide11 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide11, COLOR_LIGHT_BG)
    add_header(slide11, "Project Conclusion & Key Takeaways")

    add_card(slide11, Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.8))
    tb_c = slide11.shapes.add_textbox(Inches(1.2), Inches(2.2), Inches(10.9), Inches(4.0))
    tf_c = tb_c.text_frame
    tf_c.word_wrap = True

    p = tf_c.paragraphs[0]
    p.text = "Key Accomplishments:"
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    conclusions = [
        "End-to-End Delivery: Successfully transitioned from zero codebase to a fully trained, evaluated, and deployed ML application.",
        "Overcame Data Scarcity: Proven that Transfer Learning combined with Data Augmentation achieves 100% validation accuracy without overfitting on small samples.",
        "Production-Grade Web UI: Provided intuitive, real-time visual classification accessible to non-technical users via Streamlit.",
        "Cloud & Version Control: Enforced professional engineering standards through GitHub repository management and Streamlit Cloud live hosting."
    ]
    for c in conclusions:
        p = tf_c.add_paragraph()
        p.text = "✔  " + c
        p.font.size = Pt(14)
        p.font.color.rgb = COLOR_TEXT_DARK
        p.space_before = Pt(14)

    # =========================================================================
    # SLIDE 12: Thank You / Q&A Slide (Dark Theme)
    # =========================================================================
    slide12 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide12, COLOR_DARK_BG)

    tb_ty = slide12.shapes.add_textbox(Inches(1.5), Inches(2.0), Inches(10.3), Inches(3.5))
    tf_ty = tb_ty.text_frame
    tf_ty.word_wrap = True

    p = tf_ty.paragraphs[0]
    p.text = "Thank You!"
    p.font.size = Pt(54)
    p.font.bold = True
    p.font.color.rgb = COLOR_WHITE

    p = tf_ty.add_paragraph()
    p.text = "Open for Questions and Discussion"
    p.font.size = Pt(22)
    p.font.color.rgb = COLOR_ACCENT
    p.space_before = Pt(14)

    p = tf_ty.add_paragraph()
    p.text = "\nAbhishek Kushwaha\nB.Tech CSE (Cloud Computing & Machine Learning) • Section 4-A\nGitHub: https://github.com/Abhish13405/pet-image-classification"
    p.font.size = Pt(15)
    p.font.color.rgb = RGBColor(203, 213, 225)
    p.space_before = Pt(20)

    # Save presentation
    prs.save(OUTPUT_PPTX)
    print(f"Presentation successfully created at: {OUTPUT_PPTX}")


if __name__ == "__main__":
    create_presentation()
