import time

import streamlit as st
import torch
from model_utils import (
    GlossToVietnameseTranslator,
    get_available_models,
    get_model_display_names,
    get_model_options,
)


def create_stage_card(stage_num, stage_name, content="", status="waiting"):
    """Create a dynamic stage card with status indicators"""
    status_colors = {
        "waiting": "🔘",
        "processing": "🔄",
        "success": "✅",
        "error": "❌",
    }

    status_indicator = status_colors.get(status, "⏳")

    if status == "waiting":
        st.markdown(
            f"""
        <div style="border: 2px dashed #cccccc; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #f8f9fa;">
            <h4 style="margin: 0; color: #6c757d;">{status_indicator} Stage {stage_num}: {stage_name}</h4>
            <div style="margin: 10px 0; padding: 15px; background-color: white; border-radius: 8px; min-height: 50px;">
                <em style="color: #6c757d;">Waiting for previous stage...</em>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    elif status == "processing":
        st.markdown(
            f"""
        <div style="border: 2px solid #ffc107; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #fff3cd;">
            <h4 style="margin: 0; color: #856404;">{status_indicator} Stage {stage_num}: {stage_name}</h4>
            <div style="margin: 10px 0; padding: 15px; background-color: white; border-radius: 8px; min-height: 50px;">
                <em style="color: #856404;">Processing...</em>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    elif status == "success":
        st.markdown(
            f"""
        <div style="border: 2px solid #28a745; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #d4edda;">
            <h4 style="margin: 0; color: #155724;">{status_indicator} Stage {stage_num}: {stage_name}</h4>
            <div style="margin: 10px 0; padding: 15px; background-color: white; border-radius: 8px; min-height: 50px;">
                <strong style="color: #333; font-size: 16px;">{content}</strong>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    elif status == "error":
        st.markdown(
            f"""
        <div style="border: 2px solid #dc3545; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #f8d7da;">
            <h4 style="margin: 0; color: #721c24;">{status_indicator} Stage {stage_num}: {stage_name}</h4>
            <div style="margin: 10px 0; padding: 15px; background-color: white; border-radius: 8px; min-height: 50px;">
                <strong style="color: #dc3545;">{content}</strong>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )


def main():
    st.set_page_config(
        page_title="German Gloss to Vietnamese Translator",
        page_icon="🤟",
        layout="wide",
    )

    # Header with gradient background
    st.markdown(
        """
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 15px; margin-bottom: 30px;">
        <h1 style="color: white; text-align: center; margin: 0; font-size: 2.5em;">🤟 German Gloss → Vietnamese Translator</h1>
        <p style="color: #f0f0f0; text-align: center; margin: 10px 0 0 0; font-size: 1.2em;">AI-Powered Sign Language Translation Pipeline</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar for model selection
    st.sidebar.title("🤖 Model Configuration")
    st.sidebar.markdown("---")

    # Get available models
    all_models = get_model_options()
    available_models = get_available_models()
    display_names = get_model_display_names()

    # Show warning if some models are unavailable
    unavailable_models = [model for model in all_models if not model["available"]]
    if unavailable_models:
        with st.sidebar.expander("⚠️ Unavailable Models", expanded=False):
            for model in unavailable_models:
                st.sidebar.error(f"❌ {model['display_name']}")
                st.sidebar.caption(f"Missing files in: {model['model_dir']}")

    # Check if we have any available models
    if not available_models:
        st.sidebar.error("❌ No models available! Please check your model directories.")
        st.error(
            "🚫 No translation models found. Please ensure model files are properly installed."
        )
        st.stop()

    # Initialize session state for model selection
    if "selected_model_index" not in st.session_state:
        st.session_state.selected_model_index = 0

    # Make sure the selected index is valid for available models
    if st.session_state.selected_model_index >= len(available_models):
        st.session_state.selected_model_index = 0

    # Model selection dropdown (only available models)
    selected_display_name = st.sidebar.selectbox(
        "Select Translation Model:",
        options=display_names,
        index=st.session_state.selected_model_index,
        help="Choose the model for translation. Direct models translate gloss→Vietnamese directly, while two-stage models go through German first.",
        key="model_selector",
    )

    # Get the selected model index for available models
    new_model_index = display_names.index(selected_display_name)

    # Check if model has changed
    model_changed = new_model_index != st.session_state.selected_model_index

    if model_changed:
        st.session_state.selected_model_index = new_model_index
        # Clear existing translator to force reload
        if "translator" in st.session_state:
            # Cleanup the existing model
            try:
                st.session_state.translator.cleanup_memory()
            except:
                pass
            del st.session_state.translator

        # Clear all translation states
        st.session_state.current_stage = 0
        st.session_state.stage_results = {}
        st.session_state.is_translating = False

        st.sidebar.success("✅ Model changed! Reloading...")
        st.rerun()

    # Show current model info in sidebar
    current_model = available_models[st.session_state.selected_model_index]
    st.sidebar.markdown("### 📊 Current Model Info")
    st.sidebar.markdown(f"**Base Model:** {current_model['model_name']}")
    st.sidebar.markdown(
        f"**Translation Type:** {'Direct' if current_model['is_direct'] else 'Two-stage'}"
    )
    st.sidebar.markdown(f"**Model Directory:** {current_model['model_dir']}")

    if current_model["is_direct"]:
        st.sidebar.info("🚀 Direct translation: Gloss → Vietnamese")
    else:
        st.sidebar.info("🔄 Two-stage translation: Gloss → German → Vietnamese")

    # Initialize the translator with selected model
    if "translator" not in st.session_state:
        selected_model_config = available_models[st.session_state.selected_model_index]

        with st.spinner("🔄 Loading AI model... This may take a moment."):
            try:
                st.session_state.translator = GlossToVietnameseTranslator(
                    selected_model_config
                )

                # Display model info
                is_direct = st.session_state.translator.is_direct
                base_model = st.session_state.translator.base_model_name

                if is_direct:
                    st.success(
                        f"✅ Model loaded! Using direct translation (Base: {base_model})"
                    )
                else:
                    st.success(
                        f"✅ Model loaded! Using 2-stage translation (Base: {base_model})"
                    )

            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
                st.stop()

    # Initialize session state for dynamic updates
    if "current_stage" not in st.session_state:
        st.session_state.current_stage = 0
    if "stage_results" not in st.session_state:
        st.session_state.stage_results = {}
    if "is_translating" not in st.session_state:
        st.session_state.is_translating = False

    # Input section - single column layout
    st.markdown("### 📝 Enter German Sign Language Gloss")
    gloss_input = st.text_area(
        "Type your gloss tokens here:",
        placeholder="Example: ICH HEUTE WETTER GUT FINDEN",
        height=120,
        help="Enter German sign language gloss tokens separated by spaces",
        key="gloss_input",
    )

    # Translation controls
    col1, col2 = st.columns([3, 1])
    with col1:
        translate_btn = st.button(
            "🔄 Start Translation",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.is_translating,
            key="translate_btn",
        )
    with col2:
        clear_btn = st.button(
            "🧹 Clear All",
            use_container_width=True,
            disabled=st.session_state.is_translating,
            key="clear_all_btn",
        )

    if clear_btn:
        st.session_state.current_stage = 0
        st.session_state.stage_results = {}
        st.session_state.is_translating = False
        st.rerun()

    # Translation Pipeline Section
    st.markdown("---")

    # Get translation method info
    is_direct = st.session_state.translator.is_direct

    if is_direct:
        st.markdown("## 🚀 Direct Translation Pipeline")
        st.info("🔄 This model translates directly from German gloss to Vietnamese")

        # Single stage for direct translation
        stage1_container = st.container()

        with stage1_container:
            if st.session_state.current_stage >= 1:
                if st.session_state.current_stage == 1:
                    create_stage_card(
                        1, "Gloss → Vietnamese Translation", status="processing"
                    )
                elif "vietnamese_sentence" in st.session_state.stage_results:
                    create_stage_card(
                        1,
                        "Gloss → Vietnamese Translation",
                        st.session_state.stage_results["vietnamese_sentence"],
                        "success",
                    )
                else:
                    create_stage_card(
                        1,
                        "Gloss → Vietnamese Translation",
                        "Failed to generate Vietnamese translation",
                        "error",
                    )
            else:
                create_stage_card(1, "Gloss → Vietnamese Translation", status="waiting")

    else:
        st.markdown("## 🔄 Two-Stage Translation Pipeline")
        st.info("🔄 This model uses 2 stages: Gloss → German → Vietnamese")

        # Stage containers for two-stage translation
        stage1_container = st.container()
        stage2_container = st.container()

        # Show pipeline stages
        with stage1_container:
            if st.session_state.current_stage >= 1:
                if st.session_state.current_stage == 1:
                    create_stage_card(1, "Gloss → German Sentence", status="processing")
                elif "german_sentence" in st.session_state.stage_results:
                    create_stage_card(
                        1,
                        "Gloss → German Sentence",
                        st.session_state.stage_results["german_sentence"],
                        "success",
                    )
                else:
                    create_stage_card(
                        1,
                        "Gloss → German Sentence",
                        "Failed to generate German sentence",
                        "error",
                    )
            else:
                create_stage_card(1, "Gloss → German Sentence", status="waiting")

        with stage2_container:
            if st.session_state.current_stage >= 2:
                if st.session_state.current_stage == 2:
                    create_stage_card(
                        2, "German → Vietnamese Translation", status="processing"
                    )
                elif "vietnamese_sentence" in st.session_state.stage_results:
                    create_stage_card(
                        2,
                        "German → Vietnamese Translation",
                        st.session_state.stage_results["vietnamese_sentence"],
                        "success",
                    )
                else:
                    create_stage_card(
                        2,
                        "German → Vietnamese Translation",
                        "Failed to translate to Vietnamese",
                        "error",
                    )
            else:
                create_stage_card(
                    2, "German → Vietnamese Translation", status="waiting"
                )

    # Process translation when button is clicked
    if translate_btn:
        if gloss_input.strip():
            st.session_state.is_translating = True
            st.session_state.current_stage = 1
            st.session_state.stage_results = {}

            # Force refresh to show stage 1 processing
            st.rerun()
        else:
            st.warning("⚠️ Please enter some German gloss text to translate.")

    # Process stages dynamically
    if st.session_state.is_translating and st.session_state.current_stage == 1:
        time.sleep(1)  # Show processing state

        try:
            # Use the main translation function
            result = st.session_state.translator.translate_gloss_to_vietnamese(
                gloss_input.strip()
            )

            if result["error"]:
                st.session_state.stage_results["error"] = result["error"]
                st.session_state.current_stage = -1
                st.rerun()
            else:
                # Store results
                st.session_state.stage_results.update(result)

                if result["method"] == "direct":
                    # Direct method - completed in one stage
                    st.session_state.current_stage = 3
                    st.session_state.is_translating = False
                else:
                    # Two-stage method - go to stage 2
                    st.session_state.current_stage = 2
                st.rerun()

        except Exception as e:
            st.session_state.stage_results["error"] = str(e)
            st.session_state.current_stage = -1
            st.rerun()

    elif st.session_state.is_translating and st.session_state.current_stage == 2:
        # This only happens for two-stage method
        time.sleep(1)  # Show processing state

        # For two-stage, stage 2 is already handled in stage 1
        # Just mark as completed
        st.session_state.current_stage = 3
        st.session_state.is_translating = False
        st.rerun()

    # Model Information Section - moved from sidebar to expander at bottom
    st.markdown("---")
    with st.expander("ℹ️ Model Information & Usage"):
        current_model = available_models[st.session_state.selected_model_index]

        st.markdown(
            f"""
        ### 🔄 Current Translation Pipeline:
        **Selected Model:** {current_model['display_name']}
        
        1. **🧹 Input Processing**: Clean & process German gloss tokens
        2. **🤖 Model Translation**: {current_model['model_name']}
           - {'Direct Gloss → Vietnamese' if current_model['is_direct'] else 'Gloss → German sentence'}
        3. **🌐 Post-processing**: {'Clean Vietnamese output' if current_model['is_direct'] else 'Google Translate: German → Vietnamese'}
        
        ### 📊 Model Specifications:
        - **Dataset**: PHOENIX-2014-T (German Sign Language)
        - **Base Model**: {current_model['model_name']}
        - **Architecture**: Transformer with LoRA fine-tuning
        - **Translation Method**: {'Direct (1-stage)' if current_model['is_direct'] else 'Two-stage (via German)'}
        - **Languages**: German Sign Language Gloss → {'Vietnamese (direct)' if current_model['is_direct'] else 'German → Vietnamese'}
        
        ### 💡 Example Inputs:
        - `ICH HEUTE WETTER GUT FINDEN` (I find today's weather good)
        - `MORGEN REGEN STARK KOMMEN` (Tomorrow strong rain will come)
        - `SONNE SCHEINEN HELL TAG` (Sun shines bright day)
        - `WIND KALT WINTER ZEIT` (Wind cold winter time)
        
        ### 🎯 Usage Tips:
        - Enter gloss tokens separated by spaces
        - Use uppercase letters for better recognition
        - Keep sentences simple and weather-related for best results
        - Use the sidebar to switch between different models
        - Clear GPU memory if you experience out-of-memory errors
        """
        )


if __name__ == "__main__":
    main()
