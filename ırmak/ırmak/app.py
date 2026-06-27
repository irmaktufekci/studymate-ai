from __future__ import annotations

import os
import random
import json
import re
from datetime import datetime
from datetime import timezone
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import streamlit as st

from studymate.config import ensure_data_dirs
from studymate.document_loader import SUPPORTED_EXTENSIONS, extract_text
from studymate.rag import (
    answer_question,
    build_vector_store,
    extract_glossary,
    split_text,
    summarize_text,
)
from studymate.storage import delete_document, load_records, save_records, save_upload


st.set_page_config(
    page_title="StudyMate AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    ensure_data_dirs()
    _inject_css()

    records = load_records()

    with st.sidebar:
        selected_doc_id = _sidebar(records)

    _topbar(records)

    if not records:
        _empty_state()
        return

    if not selected_doc_id:
        selected_doc_id = next(iter(records.keys()))

    record = load_records()[selected_doc_id]
    _document_dashboard(record)


def _sidebar(records):
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-mark">📚</div>
            <div>
                <div class="brand-name">StudyMate AI</div>
                <div class="brand-caption">Akıllı Çalışma Asistanın</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-section-title">Belge Yükle</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "PDF veya TXT",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files and st.button("🚀 Analizi Başlat", type="primary", use_container_width=True):
        for uploaded_file in uploaded_files:
            _process_upload(uploaded_file)
        st.rerun()

    total_size = sum(record.size_bytes for record in records.values())
    st.markdown(
        f"""
        <div class="side-stats">
            <div><span>{len(records)}</span><small>Belge</small></div>
            <div><span>{_format_size(total_size)}</span><small>Depolama</small></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-section-title">Aktif Belge</div>', unsafe_allow_html=True)
    return _document_selector(records)


def _topbar(records) -> None:
    total_chunks = sum(record.chunk_count for record in records.values())
    glossary_terms = sum(len(record.glossary or []) for record in records.values())
    total_size = sum(record.size_bytes for record in records.values())

    st.markdown(
        f"""
        <section class="studio">
            <div class="studio-top">
                <div>
                    <div class="eyebrow">✨ StudyMate AI</div>
                    <h1>Belgelerini yükle, özetini al, sorularını sor.</h1>
                </div>
                <div class="studio-status">
                    <span class="pulse"></span>
                    <strong>Hazır</strong>
                    <small>Yapay Zeka Destekli</small>
                </div>
            </div>
            <div class="studio-body">
                <div class="studio-copy">
                    <p>
                        PDF ve TXT belgelerini analiz et, önemli kavramları çıkar,
                        kişisel notlarını tut ve belgeye doğrudan soru sor.
                    </p>
                    <div class="hero-chips">
                        <span>📄 PDF Analizi</span>
                        <span>📖 Kavram Sözlüğü</span>
                        <span>📝 Not Defteri</span>
                        <span>💬 Belge Sohbeti</span>
                    </div>
                </div>
                <div class="workflow-card">
                    <div><b>01</b><span>Yükle</span></div>
                    <div><b>02</b><span>İndeksle</span></div>
                    <div><b>03</b><span>Özetle</span></div>
                    <div><b>04</b><span>Sor</span></div>
                </div>
            </div>
        </section>
        <section class="kpi-grid">
            <div class="kpi-card metric-docs"><span>📁 Belgeler</span><strong>{len(records)}</strong><small>Yüklenen kaynak</small></div>
            <div class="kpi-card metric-chunks"><span>🧩 Parça</span><strong>{total_chunks}</strong><small>Aranabilir parça</small></div>
            <div class="kpi-card metric-terms"><span>📚 Sözlük</span><strong>{glossary_terms}</strong><small>Çıkarılan terim</small></div>
            <div class="kpi-card metric-size"><span>💾 Boyut</span><strong>{_format_size(total_size)}</strong><small>Dosya boyutu</small></div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _process_upload(uploaded_file) -> None:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        st.error(f"{uploaded_file.name} desteklenmiyor.")
        return

    with st.spinner(f"{uploaded_file.name} işleniyor..."):
        record = save_upload(uploaded_file, uploaded_file.name)
        text = extract_text(Path(record.stored_path))
        if not text.strip():
            st.error(f"{uploaded_file.name} içinden metin çıkarılamadı.")
            return

        documents = split_text(text, record.doc_id, record.filename)
        build_vector_store(documents, record.doc_id)

        record.chunk_count = len(documents)
        
        try:
            record.summary = summarize_text(text)
        except Exception as e:
            st.warning("Özet oluşturulurken API limitine ulaşıldı veya hata oluştu. Daha sonra yeniden oluşturabilirsiniz.")
            record.summary = "**Hata:** Yapay zeka modeli günlük kullanım limitine (Rate Limit) ulaştığı için özet oluşturulamadı. Lütfen daha sonra 'Özeti Yeniden Üret' butonunu kullanarak tekrar deneyin."
            
        try:
            record.glossary = extract_glossary(text)
        except Exception as e:
            st.warning("Kavram sözlüğü oluşturulurken API limitine ulaşıldı veya hata oluştu.")
            record.glossary = []

        records = load_records()
        records[record.doc_id] = record
        save_records(records)

    st.success(f"{uploaded_file.name} hazır.")


def _document_selector(records):
    if not records:
        st.markdown(
            """
            <div class="empty-mini">
                <strong>Belge Bekleniyor</strong>
                <span>Sol panelden PDF veya TXT ekleyerek analiz hattını başlat.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return None

    labels = {
        f"{record.filename}  |  {record.chunk_count} parça": doc_id
        for doc_id, record in records.items()
    }
    selected_label = st.radio("Aktif belge", list(labels.keys()), label_visibility="collapsed")

    if st.button("🗑️ Seçili Belgeyi Sil", use_container_width=True):
        doc_to_delete = labels[selected_label]
        # Guarantee it disappears from UI immediately
        records.pop(doc_to_delete, None)
        save_records(records)
        try:
            delete_document(doc_to_delete)
        except Exception:
            pass # Swallow any lingering PermissionErrors if storage.py wasn't reloaded
        st.rerun()

    return labels[selected_label]


def _document_dashboard(record) -> None:
    created = _format_date(record.created_at)
    glossary_count = len(record.glossary or [])

    st.markdown(
        f"""
        <section class="document-header">
            <div>
                <div class="eyebrow">Aktif Doküman</div>
                <h2>{record.filename}</h2>
                <p>{record.file_type} kaynak · {created} · {record.doc_id}</p>
            </div>
            <div class="doc-actions">
                <span>{record.chunk_count} chunk</span>
                <span>{glossary_count} kavram</span>
                <span>{_format_size(record.size_bytes)}</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    tab_summary, tab_glossary, tab_notes, tab_chat = st.tabs(
        ["📝 Özet", "📖 Kavram Sözlüğü", "✍️ Notlar", "🤖 StudyMate AI"]
    )

    with tab_summary:
        st.markdown('<div class="section-title">Çalışma Özeti</div>', unsafe_allow_html=True)
        if st.button("🔄 Özeti Yeniden Üret"):
            _reanalyze_document(record.doc_id)
            st.rerun()
        if record.summary:
            st.markdown(f"<div class='premium-summary-card'>\n\n{record.summary}\n\n</div>", unsafe_allow_html=True)
        else:
            with st.container(border=True):
                st.markdown("Özet henüz oluşturulmadı.")

    with tab_glossary:
        glossary = record.glossary or []
        st.markdown('<div class="section-title" style="margin-top: 0; padding-top: 0;">Kavram Sözlüğü</div>', unsafe_allow_html=True)
        if glossary:
            _glossary_grid(glossary)
        else:
            st.info("Kavram sözlüğü henüz oluşturulmadı.")

    with tab_notes:
        _notes_workspace(record)

    with tab_chat:
        _chat_workspace(record)





def _render_chat_message(content: str, msg_idx: int, role: str) -> None:
    quiz_match = re.search(r"<quiz>(.*?)</quiz>", content, re.DOTALL)
    if quiz_match:
        before = content[:quiz_match.start()].strip()
        if before:
            st.markdown(before)
            
        try:
            quiz_data = json.loads(quiz_match.group(1).strip())
            st.markdown("### 📝 Bilgi Testi")
            for i, q in enumerate(quiz_data):
                with st.container(border=True):
                    st.markdown(f"**Soru {i+1}:** {q.get('soru', '')}")
                    options = q.get("secenekler", [])
                    correct = q.get("cevap", "")
                    
                    state_key = f"quiz_radio_{msg_idx}_{i}"
                    user_choice = st.radio("Seçenekler", options, key=state_key, index=None, label_visibility="collapsed")
                    
                    if user_choice:
                        # Yanıt kontrolü (A) vs. A) formatında olabileceği için baş harflere veya eşleşmeye bakalım)
                        is_correct = user_choice.strip() == correct.strip()
                        if not is_correct and len(correct) > 0 and len(user_choice) > 0:
                            is_correct = user_choice[0].lower() == correct[0].lower()
                            
                        if is_correct:
                            st.success("✅ Doğru!")
                        else:
                            st.error(f"❌ Yanlış. Doğru cevap: **{correct}**")
                            
        except Exception as e:
            st.error(f"Test oluşturulurken hata: {str(e)}")
            st.markdown(content)
            
        after = content[quiz_match.end():].strip()
        if after:
            st.markdown(after)
    else:
        st.markdown(content)

def _chat_workspace(record) -> None:
    st.markdown(
        """
        <div class="ai-bubble">
            <div class="ai-bubble-icon">🤖</div>
            <div class="ai-bubble-content">
                <strong>StudyMate AI</strong>
                <span>Merhaba! Yüklediğin belgeyi inceledim. Bana metinle ilgili dilediğin soruyu sorabilirsin, parçalara dayanarak cevaplayacağım.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    question = st.chat_input("Bu belge hakkında ne öğrenmek istiyorsun?")
    queued_q = st.session_state.pop(f"queued_question_{record.doc_id}", None)
    if queued_q:
        question = queued_q

    if "messages" not in st.session_state:
        st.session_state.messages = {}
    history = st.session_state.messages.setdefault(record.doc_id, [])

    for idx, message in enumerate(history):
        with st.chat_message(message["role"]):
            _render_chat_message(message["content"], idx, message["role"])

    if question:
        history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            _render_chat_message(question, len(history)-1, "user")

        with st.chat_message("assistant"):
            with st.spinner("Belgede aranıyor..."):
                answer, docs = answer_question(record.doc_id, question)
                _render_chat_message(answer, len(history), "assistant")
        history.append({"role": "assistant", "content": answer})

    # PROMPTS AT THE BOTTOM
    pool = [
        "💡 Bu dokümanın ana argümanı nedir?",
        "🔑 Sınav için hangi kavramlara odaklanmalıyım?",
        "🔍 Metindeki yöntem ve bulguları özetle.",
        "📝 Belgedeki bilgilerle 5 soruluk test hazırla.",
        "📊 Belgedeki sayısal verileri ve bulguları çıkar.",
        "🎯 Yazarın çözmeye çalıştığı asıl problem nedir?",
        "⚖️ Metindeki ana tartışma noktaları nelerdir?",
        "⚙️ Kullanılan metodoloji ve yaklaşımı açıkla.",
        "📌 Sonuç bölümündeki en önemli 3 madde nedir?"
    ]
    
    state_key = f"prompts_{record.doc_id}"
    if state_key not in st.session_state:
        st.session_state[state_key] = random.sample(pool, 3)
        
    st.markdown("<br><p style='font-size:0.9rem; color:var(--text-secondary); margin-bottom: 0.5rem;'>Hızlı Sorular:</p>", unsafe_allow_html=True)
    
    # Custom CSS for these specific buttons
    st.markdown("""
    <style>
    div[data-testid="stTabContent"]:nth-of-type(4) div[data-testid="column"] button {
        white-space: normal !important;
        height: auto !important;
        min-height: 80px !important;
        padding: 1rem !important;
        border-radius: 12px !important;
        background-color: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-color) !important;
        font-size: 0.95rem !important;
        text-align: left !important;
        box-shadow: var(--shadow-sm) !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stTabContent"]:nth-of-type(4) div[data-testid="column"] button:hover {
        border-color: var(--primary-color) !important;
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-md) !important;
    }
    div[data-testid="stTabContent"]:nth-of-type(4) div[data-testid="column"] button p {
        margin: 0 !important;
        text-align: left !important;
        line-height: 1.4 !important;
    }
            /* AI BUBBLE AND CHAT INPUT STYLING */
        .ai-bubble {
            display: flex;
            align-items: center;
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            padding: 1rem 1.5rem;
            border-radius: 20px 20px 20px 4px;
            color: white;
            box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.4);
            margin-bottom: 2rem;
            animation: float 3s ease-in-out infinite;
            gap: 1rem;
        }
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-5px); }
            100% { transform: translateY(0px); }
        }
        .ai-bubble-icon {
            font-size: 2rem;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
        .ai-bubble-content strong {
            display: block;
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
            color: white;
        }
        .ai-bubble-content span {
            font-size: 0.95rem;
            opacity: 0.95;
            line-height: 1.4;
            color: white;
        }
        
        /* Modern Chat Input Styling override */
        div[data-testid="stChatInput"] {
            border-radius: 12px !important;
            border: 1px solid var(--border-color) !important;
            background-color: var(--card-bg) !important;
            box-shadow: var(--shadow-sm) !important;
            transition: all 0.2s ease;
        }
        div[data-testid="stChatInput"]:focus-within {
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 2px rgba(99,102,241,0.2) !important;
        }
        
                /* GLOSSARY CARD STYLING */
        .glossary-card {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.25rem;
            height: 100%;
            min-height: 140px;
            box-shadow: var(--shadow-sm);
            transition: all 0.3s ease;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            position: relative;
            overflow: hidden;
        }
        .glossary-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: linear-gradient(to bottom, #6366f1, #a855f7);
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        .glossary-card:hover {
            transform: translateY(-4px);
            box-shadow: var(--shadow-md);
            border-color: rgba(99, 102, 241, 0.3);
        }
        .glossary-card:hover::before {
            opacity: 1;
        }
        .glossary-term {
            font-weight: 700;
            font-size: 1.05rem;
            color: var(--primary-color);
            line-height: 1.3;
        }
        .glossary-desc {
            font-size: 0.95rem;
            color: var(--text-secondary);
            line-height: 1.5;
        }

                /* PREMIUM SUMMARY CARD STYLING */
        .premium-summary-card {
            background-color: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 2.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            margin-top: 1.5rem;
            color: var(--ink-secondary);
        }
        .premium-summary-card h1 {
            color: var(--primary);
            font-size: 2.2rem;
            margin-bottom: 1.2rem;
            border-bottom: 2px solid var(--primary-light);
            padding-bottom: 0.8rem;
            font-weight: 800;
        }
        .premium-summary-card h2 {
            color: var(--ink-primary);
            font-size: 1.6rem;
            margin-top: 2.5rem;
            margin-bottom: 1rem;
            font-weight: 700;
        }
        .premium-summary-card p {
            line-height: 1.8;
            font-size: 1.05rem;
            color: var(--ink-secondary);
            margin-bottom: 1.2rem;
        }
        .premium-summary-card ul {
            padding-left: 1.5rem;
            margin-bottom: 1.5rem;
        }
        .premium-summary-card li {
            margin-bottom: 0.6rem;
            line-height: 1.7;
            color: var(--ink-secondary);
        }
        .premium-summary-card strong {
            color: var(--ink-primary);
            font-weight: 700;
        }

        </style>
    """, unsafe_allow_html=True)

    cols = st.columns(3)
    for i, prompt_text in enumerate(st.session_state[state_key]):
        with cols[i]:
            if st.button(prompt_text, key=f"pbtn_{record.doc_id}_{i}", use_container_width=True):
                st.session_state[f"queued_question_{record.doc_id}"] = prompt_text
                st.rerun()
                
    if st.button("🔄 Farklı Sorular Öner", key=f"refresh_pbtn_{record.doc_id}"):
        st.session_state[state_key] = random.sample(pool, 3)
        st.rerun()


def _notes_workspace(record) -> None:
    st.markdown(
        """
        <div class="notes-intro">
            <strong>Kişisel Notlar</strong>
            <span>Bu alan aktif belgeye bağlıdır; yazdıklarınız dosya değiştiğinde kaybolmaz.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    notes = _normalized_notes(record.notes)
    if f"active_note_{record.doc_id}" not in st.session_state:
        st.session_state[f"active_note_{record.doc_id}"] = notes[0]["id"] if notes else ""

    col_list, col_editor = st.columns([1, 2.4], gap="large")

    with col_list:
        st.markdown('<div class="section-title">Not Sayfaları</div>', unsafe_allow_html=True)
        if st.button("➕ Yeni Not Sayfası", type="primary", use_container_width=True):
            new_note = _create_note(record.doc_id)
            st.session_state[f"active_note_{record.doc_id}"] = new_note["id"]
            st.rerun()

        notes = _normalized_notes(load_records()[record.doc_id].notes)
        if notes:
            note_options = {
                f"{index}. {note.get('title') or 'Başlıksız'}": note["id"]
                for index, note in enumerate(notes, start=1)
            }
            current_id = st.session_state.get(f"active_note_{record.doc_id}", notes[0]["id"])
            labels = list(note_options.keys())
            current_label = next(
                (label for label, note_id in note_options.items() if note_id == current_id),
                labels[0],
            )
            selected_label = st.radio(
                "Notlar",
                labels,
                index=labels.index(current_label),
                label_visibility="collapsed",
            )
            st.session_state[f"active_note_{record.doc_id}"] = note_options[selected_label]
        else:
            st.info("Henüz not sayfası yok.")

    notes = _normalized_notes(load_records()[record.doc_id].notes)
    active_id = st.session_state.get(f"active_note_{record.doc_id}", "")
    active_note = next((note for note in notes if note["id"] == active_id), None)

    with col_editor:
        if not active_note:
            st.markdown(
                """
                <div class="empty-mini">
                    <strong>Not sayfası seçilmedi</strong>
                    <span>Yeni not sayfasi olusturarak bu belge hakkindaki düşüncelerini kaydet.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        title = st.text_input(
            "Not Başlığı",
            value=active_note.get("title", "Başlıksız"),
            key=f"note_title_{record.doc_id}_{active_note['id']}",
        )
        content = st.text_area(
            "Notlarını Yaz",
            value=active_note.get("content", ""),
            height=360,
            placeholder=(
                "Örnek:\n"
                "- Bu belgenin ana fikri...\n"
                "- Sinav icin tekrar etmem gereken kavramlar...\n"
                "- Hocaya sorulacak soru..."
            ),
            key=f"note_content_{record.doc_id}_{active_note['id']}",
            label_visibility="collapsed",
        )

        updated_at = active_note.get("updated_at") or "Henüz kaydedilmedi"
        st.caption(f"Son kayıt: {updated_at} · {len(content)} karakter")

        col_save, col_clear, col_delete = st.columns(3)
        with col_save:
            if st.button("💾 Kaydet", type="primary", use_container_width=True):
                _save_note(record.doc_id, active_note["id"], title, content)
                st.success("Not kaydedildi.")
        with col_clear:
            if st.button("🧹 İçeriği Temizle", use_container_width=True):
                _save_note(record.doc_id, active_note["id"], title, "")
                st.rerun()
        with col_delete:
            if st.button("🗑️ Notu Sil", use_container_width=True):
                _delete_note(record.doc_id, active_note["id"])
                st.rerun()


def _normalized_notes(notes) -> list[dict[str, str]]:
    if isinstance(notes, list):
        return notes
    if isinstance(notes, str) and notes.strip():
        return [{"id": "note_1", "title": "Not 1", "content": notes, "updated_at": ""}]
    return []


def _create_note(doc_id: str) -> dict[str, str]:
    records = load_records()
    record = records[doc_id]
    notes = _normalized_notes(record.notes)
    note = {
        "id": f"note_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
        "title": f"Not {len(notes) + 1}",
        "content": "",
        "updated_at": _now_label(),
    }
    notes.append(note)
    record.notes = notes
    records[doc_id] = record
    save_records(records)
    return note


def _save_note(doc_id: str, note_id: str, title: str, content: str) -> None:
    records = load_records()
    record = records[doc_id]
    notes = _normalized_notes(record.notes)
    for note in notes:
        if note["id"] == note_id:
            note["title"] = title.strip() or "Başlıksız"
            note["content"] = content
            note["updated_at"] = _now_label()
            break
    record.notes = notes
    records[doc_id] = record
    save_records(records)


def _delete_note(doc_id: str, note_id: str) -> None:
    records = load_records()
    record = records[doc_id]
    notes = [note for note in _normalized_notes(record.notes) if note["id"] != note_id]
    record.notes = notes
    records[doc_id] = record
    save_records(records)
    st.session_state[f"active_note_{doc_id}"] = notes[0]["id"] if notes else ""


def _now_label() -> str:
    return datetime.now().strftime("%d.%m.%Y %H:%M")


def _file_manager() -> None:
    records = load_records()
    table = [
        {
            "Belge": item.filename,
            "Tür": item.file_type,
            "Parça": item.chunk_count,
            "Boyut": _format_size(item.size_bytes),
            "Oluşturulma": _format_date(item.created_at),
            "ID": doc_id,
        }
        for doc_id, item in records.items()
    ]
    st.markdown('<div class="section-title">Yerel Kütüphane</div>', unsafe_allow_html=True)
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption("Dosyalar data/uploads altında, FAISS indeksleri data/indexes altında tutulur.")


def _reanalyze_document(doc_id: str) -> None:
    records = load_records()
    record = records[doc_id]
    with st.spinner("Özet ve kavram sözlüğü yeniden üretiliyor..."):
        text = extract_text(Path(record.stored_path))
        try:
            record.summary = summarize_text(text)
        except Exception as e:
            st.error("Özet oluşturulurken API limitine ulaşıldı veya hata oluştu.")
            record.summary = "**Hata:** Yapay zeka modeli günlük kullanım limitine (Rate Limit) ulaştığı için özet oluşturulamadı. Lütfen daha sonra 'Özeti Yeniden Üret' butonunu kullanarak tekrar deneyin."
            
        try:
            record.glossary = extract_glossary(text)
        except Exception as e:
            st.error("Kavram sözlüğü oluşturulurken API limitine ulaşıldı veya hata oluştu.")
            record.glossary = []
            
        records[doc_id] = record
        save_records(records)


def _glossary_grid(glossary: list[dict[str, str]]) -> None:
    cols = st.columns(3)
    for index, item in enumerate(glossary):
        term = item.get("term", "")
        description = item.get("description", "")
        
        if term.lower() in ["terim", "kavram"] and "-" in description:
            parts = description.split("-", 1)
            term = parts[0].strip()
            description = parts[1].strip()
            
        with cols[index % 3]:
            st.markdown(f"""
            <div class="glossary-card">
                <div class="glossary-term">{term}</div>
                <div class="glossary-desc">{description}</div>
            </div>
            """, unsafe_allow_html=True)


def _empty_state() -> None:
    st.markdown(
        """
        <section class="empty-state">
            <div>
                <div class="eyebrow">Başlangıç</div>
                <h2>İlk Akademik Kaynağını Yükle</h2>
                <p>
                    StudyMate AI metni çıkarır, parçalara ayırır, FAISS indeksini oluşturur,
                    özet ve kavram sözlüğü üretir. Sonrasında belgeye doğrudan soru sorabilirsin.
                </p>
            </div>
            <div class="pipeline">
                <div><span>01</span><strong>Metin çıkar</strong><small>PDF/TXT içeriği okunur</small></div>
                <div><span>02</span><strong>Parçalara ayır</strong><small>Akademik parçalar hazırlanır</small></div>
                <div><span>03</span><strong>İndeksle</strong><small>FAISS arama alanı kurulur</small></div>
                <div><span>04</span><strong>Soru sor</strong><small>RAG ile kaynaklı yanıt alınır</small></div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    
    with st.container(border=True):
        st.markdown('<div class="section-title" style="margin-top: 0; padding-top: 0;">Dosya Yükle</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "PDF veya TXT belgelerini buraya ekle",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key="main_file_uploader",
            label_visibility="collapsed"
        )
        if uploaded_files and st.button("Belgeleri Analiz Et", type="primary", use_container_width=True):
            for uploaded_file in uploaded_files:
                _process_upload(uploaded_file)
            st.rerun()


def _format_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _format_date(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.strftime("%d.%m.%Y")
    except ValueError:
        return value[:10]


def _markdown_to_html(value: str) -> str:
    lines = []
    for line in _escape(value).splitlines():
        if line.startswith("- "):
            lines.append(f"<p class='summary-bullet'>{line}</p>")
        elif line.strip():
            lines.append(f"<p>{line}</p>")
    return "".join(lines)


def _escape(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

        /* ═══ DESIGN SYSTEM & TOKENS ═══ */
        :root {
            --bg: #f8fafc;
            --surface: #ffffff;
            --surface-hover: #f1f5f9;
            --ink-primary: #0f172a;
            --ink-secondary: #334155;
            --ink-muted: #64748b;
            --border: #e2e8f0;
            --border-hover: #cbd5e1;
            
            /* Accents */
            --primary: #6366f1;
            --primary-gradient: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
            --primary-light: rgba(99, 102, 241, 0.08);
            
            --success: #10b981;
            --success-gradient: linear-gradient(135deg, #10b981 0%, #059669 100%);
            --success-light: rgba(16, 185, 129, 0.08);
            
            --rose: #f43f5e;
            --rose-gradient: linear-gradient(135deg, #f43f5e 0%, #e11d48 100%);
            --rose-light: rgba(244, 63, 94, 0.08);
            
            --amber: #f59e0b;
            --amber-gradient: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
            --amber-light: rgba(245, 158, 11, 0.08);
            
            --sidebar-gradient: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
            
            /* Shadows & Blur */
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -4px rgba(0, 0, 0, 0.05);
            --shadow-premium: 0 20px 25px -5px rgba(99, 102, 241, 0.05), 0 8px 10px -6px rgba(99, 102, 241, 0.05);
            
            /* Additional compatibility variables */
            --card-bg: var(--surface);
            --border-color: var(--border);
            --text-color: var(--ink-secondary);
            --text-primary: var(--ink-primary);
            --text-secondary: var(--ink-muted);
            --hover-color: var(--surface-hover);
            --bg-color: var(--surface);
            --primary-color: var(--primary);

            --radius-lg: 16px;
            --radius-md: 12px;
            --radius-sm: 8px;
            --transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* ═══ GLOBAL STYLE OVERRIDES ═══ */
        *, *::before, *::after {
            font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif;
        }
        
        .stApp {
            background-color: var(--bg) !important;
            color: var(--ink-primary) !important;
        }

        /* Enforce dark text contrast on light background (stops dark mode browser plugins or streamlit theme inheritance from blowing out text) */
        .stApp p, .stApp span, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
            color: var(--ink-primary) !important;
        }

        /* Keep sidebar texts light */
        [data-testid="stSidebar"] p, 
        [data-testid="stSidebar"] span, 
        [data-testid="stSidebar"] label, 
        [data-testid="stSidebar"] li, 
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3, 
        [data-testid="stSidebar"] strong,
        [data-testid="stSidebar"] small {
            color: rgba(241, 245, 249, 0.9) !important;
        }

        .block-container {
            max-width: 1400px;
            padding: 40px 60px;
        }

        header[data-testid="stHeader"] {
            background: transparent !important;
        }

        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        .stDeployButton,
        #MainMenu {
            display: none !important;
        }

        /* ═══ MODERN SIDEBAR ═══ */
        [data-testid="stSidebar"] {
            background: var(--sidebar-gradient);
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }
        
        [data-testid="stSidebar"] * {
            color: rgba(241, 245, 249, 0.9);
        }

        [data-testid="stSidebar"] .block-container {
            padding: 32px 24px;
        }

        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 20px;
            margin-bottom: 28px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: var(--radius-md);
            backdrop-filter: blur(12px);
            box-shadow: var(--shadow-sm);
        }

        .brand-mark {
            width: 48px;
            height: 48px;
            display: grid;
            place-items: center;
            border-radius: var(--radius-sm);
            background: linear-gradient(135deg, var(--primary), #818cf8);
            font-size: 24px;
            line-height: 1;
            box-shadow: 0 8px 20px rgba(99, 102, 241, 0.3);
        }

        .brand-name {
            font-size: 19px;
            font-weight: 800;
            color: #ffffff;
            letter-spacing: -0.5px;
        }

        .brand-caption {
            font-size: 11px;
            color: rgba(241, 245, 249, 0.5);
            margin-top: 2px;
        }

        .side-section-title {
            margin: 24px 0 12px;
            font-size: 10px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: rgba(129, 140, 248, 0.8);
        }

        .side-stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-bottom: 24px;
        }

        .side-stats div {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: var(--radius-md);
            padding: 16px;
            transition: var(--transition);
        }

        .side-stats div:hover {
            background: rgba(255, 255, 255, 0.06);
            border-color: rgba(255, 255, 255, 0.1);
            transform: translateY(-2px);
        }

        .side-stats span {
            display: block;
            font-size: 24px;
            font-weight: 800;
            color: #ffffff;
        }

        .side-stats small {
            display: block;
            font-size: 11px;
            color: rgba(241, 245, 249, 0.4);
            margin-top: 4px;
        }

        .empty-mini {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: var(--radius-md);
            padding: 20px;
            text-align: center;
        }

        .empty-mini strong {
            color: #ffffff;
            font-size: 13px;
        }

        .empty-mini span {
            display: block;
            color: rgba(241, 245, 249, 0.45);
            font-size: 12px;
            margin-top: 6px;
        }

        [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
            background: rgba(255, 255, 255, 0.02) !important;
            border: 1.5px dashed rgba(99, 102, 241, 0.3) !important;
            border-radius: var(--radius-md) !important;
            transition: var(--transition) !important;
        }

        [data-testid="stSidebar"] [data-testid="stFileUploader"] section:hover {
            border-color: rgba(99, 102, 241, 0.6) !important;
            background: rgba(255, 255, 255, 0.04) !important;
        }

        [data-testid="stSidebar"] button {
            background: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            color: #ffffff !important;
            border-radius: var(--radius-md) !important;
            font-weight: 600 !important;
            transition: var(--transition) !important;
        }

        [data-testid="stSidebar"] button:hover {
            background: rgba(255, 255, 255, 0.1) !important;
            border-color: rgba(255, 255, 255, 0.15) !important;
            transform: translateY(-1px);
        }

        [data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background: var(--primary-gradient) !important;
            border: none !important;
            box-shadow: 0 4px 14px rgba(99, 102, 241, 0.3) !important;
        }

        [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
        }

        /* ═══ DESKTOP SIDEBAR ALWAYS EXPANDED ═══ */
        @media (min-width: 901px) {
            section[data-testid="stSidebar"][aria-expanded="false"] {
                transform: none !important;
                margin-left: 0 !important;
                width: 21rem !important;
                min-width: 21rem !important;
                visibility: visible !important;
                display: block !important;
            }
            [data-testid="collapsedControl"],
            [data-testid="stSidebarCollapseButton"] {
                display: none !important;
            }
        }

        @media (max-width: 900px) {
            [data-testid="collapsedControl"] {
                display: flex !important;
                visibility: visible !important;
                position: fixed !important;
                top: 16px !important;
                left: 16px !important;
                z-index: 999999 !important;
            }
            [data-testid="collapsedControl"] button {
                width: 44px !important;
                height: 44px !important;
                border-radius: 50% !important;
                background: #0f172a !important;
                color: #ffffff !important;
                box-shadow: var(--shadow-lg) !important;
            }
        }

        /* ═══ PREMIUM STUDIO HERO ═══ */
        .studio {
            position: relative;
            overflow: hidden;
            border-radius: var(--radius-lg);
            border: 1px solid var(--border);
            background: var(--surface);
            box-shadow: var(--shadow-premium);
            margin-bottom: 28px;
            transition: var(--transition);
        }

        .studio:hover {
            box-shadow: 0 25px 50px -12px rgba(99, 102, 241, 0.08);
        }

        .studio::before {
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--primary), var(--success), var(--amber), var(--rose));
        }

        .studio::after {
            content: "";
            position: absolute;
            top: 0; right: 0;
            width: 35%; height: 100%;
            background: radial-gradient(ellipse at 100% 0%, rgba(99, 102, 241, 0.05) 0%, transparent 80%);
            pointer-events: none;
        }

        .studio-top {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 32px;
            padding: 36px 36px 16px;
        }

        .studio h1 {
            margin: 0;
            font-size: 32px;
            font-weight: 800;
            color: var(--ink-primary);
            line-height: 1.2;
            letter-spacing: -1px;
        }

        .studio-body {
            display: grid;
            grid-template-columns: 1fr 380px;
            gap: 32px;
            padding: 0 36px 36px;
        }

        .studio-copy p {
            color: var(--ink-secondary);
            font-size: 15px;
            line-height: 1.75;
            margin: 0 0 20px;
        }

        .eyebrow {
            font-size: 11px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: var(--primary);
            margin-bottom: 12px;
        }

        .hero-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 20px;
        }

        .hero-chips span {
            padding: 8px 18px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 600;
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--ink-secondary);
            box-shadow: var(--shadow-sm);
            transition: var(--transition);
        }

        .hero-chips span:hover {
            border-color: var(--primary);
            background: var(--primary-light);
            color: var(--primary);
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }

        .studio-status {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 16px;
            border-radius: var(--radius-md);
            background: rgba(16, 185, 129, 0.08);
            border: 1px solid rgba(16, 185, 129, 0.15);
            color: var(--success);
            white-space: nowrap;
        }

        .studio-status strong {
            color: var(--success);
            font-size: 13px;
            font-weight: 700;
        }

        .studio-status small {
            color: rgba(5, 150, 105, 0.7);
            font-size: 11px;
            margin-left: 2px;
        }

        .pulse {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--success);
            box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.3);
            animation: pulse-glow 2s ease-in-out infinite;
        }

        @keyframes pulse-glow {
            0%, 100% { box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.3); }
            50% { box-shadow: 0 0 0 8px rgba(16, 185, 129, 0.1); }
        }

        .workflow-card {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            padding: 12px;
            border-radius: var(--radius-md);
            background: #f8fafc;
            border: 1px solid var(--border);
            box-shadow: var(--shadow-sm);
        }

        .workflow-card div {
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            min-height: 80px;
            border-radius: var(--radius-sm);
            padding: 14px;
            background: #ffffff;
            border: 1px solid var(--border);
            transition: var(--transition);
        }

        .workflow-card div:hover {
            background: var(--primary-light);
            border-color: var(--primary);
            transform: translateY(-1px);
        }

        .workflow-card b {
            color: var(--primary);
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 1px;
        }

        .workflow-card span {
            color: var(--ink-primary);
            font-weight: 700;
            font-size: 14px;
        }

        /* ═══ MODERN KPI CARDS ═══ */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 28px;
        }

        .kpi-card {
            position: relative;
            overflow: hidden;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 24px;
            box-shadow: var(--shadow-sm);
            transition: var(--transition);
        }

        .kpi-card:hover {
            transform: translateY(-4px);
            box-shadow: var(--shadow-lg);
            border-color: var(--border-hover);
        }

        .kpi-card::before {
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 4px;
            background: var(--metric-color, var(--primary));
        }

        .metric-docs   { --metric-color: var(--primary); }
        .metric-chunks { --metric-color: var(--success); }
        .metric-terms  { --metric-color: var(--amber); }
        .metric-size   { --metric-color: var(--rose); }

        .kpi-card span {
            font-size: 10px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--ink-muted);
        }

        .kpi-card strong {
            display: block;
            margin-top: 12px;
            font-size: 34px;
            font-weight: 800;
            color: var(--ink-primary);
            letter-spacing: -1px;
            line-height: 1;
        }

        .kpi-card small {
            display: block;
            margin-top: 8px;
            font-size: 13px;
            color: var(--ink-muted);
        }

        /* ═══ ACTIVE DOCUMENT HEADER ═══ */
        .document-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 20px;
            flex-wrap: wrap;
            margin: 24px 0;
            padding: 24px 32px;
            background: var(--surface);
            border: 1px solid var(--border);
            border-left: 4px solid var(--primary);
            border-radius: var(--radius-md);
            box-shadow: var(--shadow-sm);
        }

        .document-header h2 {
            margin: 0;
            font-size: 21px;
            font-weight: 800;
            letter-spacing: -0.5px;
            color: var(--ink-primary);
        }

        .document-header p {
            margin: 6px 0 0;
            color: var(--ink-muted);
            font-size: 13px;
        }

        .doc-actions {
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }

        .doc-actions span {
            padding: 8px 16px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
            background: var(--primary-light);
            color: var(--primary);
            border: 1px solid rgba(99, 102, 241, 0.15);
        }

        .doc-actions span:nth-child(2) {
            background: var(--success-light);
            color: var(--success);
            border-color: rgba(16, 185, 129, 0.15);
        }

        .doc-actions span:nth-child(3) {
            background: var(--rose-light);
            color: var(--rose);
            border-color: rgba(244, 63, 94, 0.15);
        }

        /* ═══ PILL STYLED TABS ═══ */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            padding: 6px;
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            background: #f1f5f9;
            width: fit-content;
            margin-top: 16px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: var(--radius-sm);
            height: 40px;
            padding: 8px 20px;
            background: transparent;
            color: var(--ink-muted) !important;
            font-weight: 600;
            font-size: 14px;
            transition: var(--transition);
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: var(--ink-primary) !important;
            background: rgba(255, 255, 255, 0.5) !important;
        }

        .stTabs [aria-selected="true"] {
            background: var(--primary) !important;
            color: #ffffff !important;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2) !important;
        }

        /* ═══ CONTENT CARD CONTAINERS ═══ */
        .section-title {
            margin: 24px 0 14px;
            font-size: 17px;
            font-weight: 800;
            color: var(--ink-primary);
        }

        .prose-box, .chat-intro, .notes-intro, .main-upload {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 28px;
            line-height: 1.8;
            color: var(--ink-secondary);
            box-shadow: var(--shadow-sm);
        }

        .main-upload {
            margin-top: 20px;
        }

        .summary-bullet {
            padding-left: 16px;
            border-left: 4px solid var(--primary);
            margin-bottom: 12px;
        }

        /* ═══ GLOSSARY CARDS ═══ */
        .glossary-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
        }

        .term-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-left: 4px solid var(--success);
            border-radius: var(--radius-md);
            padding: 20px;
            box-shadow: var(--shadow-sm);
            transition: var(--transition);
        }

        .term-card:hover {
            transform: translateY(-3px);
            box-shadow: var(--shadow-md);
            border-left-color: var(--primary);
            border-color: var(--border-hover);
        }

        .term-card strong {
            font-size: 15px;
            font-weight: 700;
            color: var(--ink-primary);
        }

        .term-card p {
            margin: 8px 0 0;
            color: var(--ink-muted);
            font-size: 13px;
            line-height: 1.6;
        }

        /* ═══ CHAT ELEMENTS ═══ */
        .chat-intro, .notes-intro {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            margin: 20px 0 14px;
            border-left: 4px solid var(--success);
            padding: 16px 24px;
        }

        .chat-intro span, .notes-intro span {
            color: var(--ink-muted);
            font-size: 13px;
        }

        .prompt-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 14px;
            margin-bottom: 20px;
        }

        .prompt-grid div {
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            border-left: 3px solid var(--primary);
            padding: 18px;
            background: var(--surface);
            color: var(--ink-secondary);
            font-size: 14px;
            line-height: 1.5;
            cursor: pointer;
            transition: var(--transition);
            box-shadow: var(--shadow-sm);
        }

        .prompt-grid div:hover {
            border-left-color: var(--rose);
            background: var(--rose-light);
            transform: translateY(-3px);
            box-shadow: var(--shadow-md);
            border-color: var(--border-hover);
        }

        .stChatMessage {
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            background: var(--surface) !important;
            box-shadow: var(--shadow-sm);
            margin-bottom: 12px;
            padding: 16px;
        }
        
        .stChatMessage * {
            color: var(--ink-primary) !important;
        }

        /* ═══ MODERN CHAT INPUT DESIGN (THEME ADAPTIVE) ═══ */
        [data-testid="stChatInput"] {
            background-color: var(--card-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: var(--radius-md) !important;
            padding: 8px 16px !important;
            box-shadow: var(--shadow-sm) !important;
            transition: all 0.2s ease;
        }
        
        [data-testid="stChatInput"]:focus-within {
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 2px rgba(99,102,241,0.2) !important;
        }
        
        [data-testid="stChatInput"] textarea {
            background-color: transparent !important;
            color: var(--text-color) !important;
            border: none !important;
            font-size: 14px !important;
            outline: none !important;
            box-shadow: none !important;
            resize: none !important;
            padding: 4px 0 !important;
        }

        [data-testid="stChatInput"] textarea:focus {
            color: var(--text-color) !important;
        }

        [data-testid="stChatInput"] textarea::placeholder {
            color: var(--text-secondary) !important;
        }
        
        [data-testid="stChatInput"] button {
            background-color: var(--primary) !important;
            color: #ffffff !important;
            border-radius: var(--radius-sm) !important;
            border: none !important;
            width: 34px !important;
            height: 34px !important;
            min-height: 34px !important;
            padding: 0 !important;
            display: grid !important;
            place-items: center !important;
            transition: var(--transition) !important;
            box-shadow: var(--shadow-sm) !important;
        }
        
        [data-testid="stChatInput"] button:hover {
            background-color: #4f46e5 !important;
            transform: scale(1.05) !important;
        }

        /* ═══ EMPTY STATES ═══ */
        .empty-state {
            display: grid;
            grid-template-columns: 1fr 380px;
            gap: 28px;
            align-items: stretch;
            margin-top: 24px;
        }

        .empty-state > div {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 40px;
            box-shadow: var(--shadow-sm);
        }

        .empty-state h2 {
            margin: 0;
            font-size: 24px;
            font-weight: 800;
            letter-spacing: -0.5px;
            color: var(--ink-primary);
        }

        .empty-state p {
            margin: 16px 0 0;
            color: var(--ink-muted);
            font-size: 15px;
            line-height: 1.75;
        }

        .pipeline {
            display: grid;
            gap: 12px;
        }

        .pipeline div {
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 20px;
            background: #f8fafc;
            transition: var(--transition);
        }

        .pipeline div:hover {
            background: var(--primary-light);
            border-color: rgba(99, 102, 241, 0.2);
            transform: translateX(4px);
        }

        .pipeline span {
            color: var(--primary);
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 1px;
            text-transform: uppercase;
        }

        .pipeline strong {
            display: block;
            margin-top: 6px;
            font-size: 15px;
            color: var(--ink-primary);
        }

        .pipeline small {
            display: block;
            margin-top: 4px;
            color: var(--ink-muted);
            font-size: 13px;
        }

        /* ═══ FORMS & INPUT FIELDS (THEME ADAPTIVE) ═══ */
        .stTextArea textarea, .stTextInput input {
            background-color: var(--surface) !important;
            color: var(--ink-primary) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-md) !important;
            font-size: 14px !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
            transition: var(--transition) !important;
            padding: 12px 16px !important;
        }

        .stTextArea textarea:focus, .stTextInput input:focus {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25) !important;
            background-color: var(--surface) !important;
        }

        .stTextArea textarea::placeholder, .stTextInput input::placeholder {
            color: var(--ink-muted) !important;
        }

        .stButton > button {
            border-radius: var(--radius-md);
            font-weight: 700;
            min-height: 44px;
            transition: var(--transition);
            letter-spacing: -0.1px;
            padding: 10px 24px;
            background-color: #ffffff !important;
            border: 1px solid var(--border-hover) !important;
            color: var(--ink-primary) !important;
        }

        .stButton > button:hover {
            background-color: var(--surface-hover) !important;
            border-color: var(--primary) !important;
            color: var(--primary) !important;
            transform: translateY(-1px);
        }

        .stButton > button[kind="primary"] {
            background: var(--primary-gradient) !important;
            border: none !important;
            color: #ffffff !important;
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.25) !important;
        }

        .stButton > button[kind="primary"]:hover {
            box-shadow: 0 10px 30px rgba(99, 102, 241, 0.35) !important;
            color: #ffffff !important;
            transform: translateY(-2px);
        }

        /* ═══ SIDEBAR BUTTONS (ALERT/DELETE STYLE OVERRIDES) ═══ */
        [data-testid="stSidebar"] .stButton > button:not([kind="primary"]) {
            background-color: #ef4444 !important;
            border: 1px solid #ef4444 !important;
            color: #ffffff !important;
            transition: var(--transition) !important;
            border-radius: var(--radius-md) !important;
            width: 100% !important;
            font-weight: 700 !important;
            box-shadow: 0 4px 12px rgba(239, 68, 68, 0.2) !important;
        }
 
        [data-testid="stSidebar"] .stButton > button:not([kind="primary"]):hover {
            background-color: #dc2626 !important;
            border-color: #dc2626 !important;
            color: #ffffff !important;
            box-shadow: 0 6px 16px rgba(239, 68, 68, 0.4) !important;
            transform: translateY(-1px);
        }

        [data-testid="stFileUploader"] section {
            border: 1.5px dashed var(--border-hover);
            border-radius: var(--radius-md);
            background: #f8fafc;
            padding: 24px;
            transition: var(--transition);
        }

        [data-testid="stFileUploader"] section:hover {
            border-color: var(--primary);
            background: var(--primary-light);
        }

        /* ═══ HIGH-CONTRAST FILE UPLOADER LABELS & BUTTONS ═══ */
        /* Main area (Light theme background) */
        [data-testid="stFileUploader"] section * {
            color: var(--ink-secondary) !important;
        }
        [data-testid="stFileUploader"] section button {
            color: var(--ink-primary) !important;
            background-color: #ffffff !important;
            border: 1px solid var(--border) !important;
        }
        [data-testid="stFileUploader"] section button:hover {
            background-color: var(--surface-hover) !important;
            border-color: var(--border-hover) !important;
        }

        /* Sidebar area (Dark theme background) */
        [data-testid="stSidebar"] [data-testid="stFileUploader"] section * {
            color: rgba(255, 255, 255, 0.9) !important;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploader"] section button {
            color: #ffffff !important;
            background-color: rgba(255, 255, 255, 0.08) !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploader"] section button:hover {
            background-color: rgba(255, 255, 255, 0.15) !important;
            border-color: rgba(255, 255, 255, 0.25) !important;
        }

        /* ═══ UPLOADED FILES LIST CONTRAST ═══ */
        /* Main view uploaded files (Light background) */
        [data-testid="stFileUploader"] [data-testid="stUploadedFile"],
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
            background-color: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-md) !important;
            padding: 10px 14px !important;
        }
        
        [data-testid="stFileUploader"] [data-testid="stUploadedFile"] *,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] * {
            color: var(--ink-primary) !important;
        }
        
        [data-testid="stFileUploader"] [data-testid="stUploadedFile"] svg,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] svg,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] svg {
            fill: var(--ink-muted) !important;
        }

        /* Sidebar view uploaded files (Dark background master overrides) */
        
        /* 1. Reset ALL backgrounds inside stFileUploader in the sidebar to transparent */
        [data-testid="stSidebar"] [data-testid="stFileUploader"] * {
            background-color: transparent !important;
            color: #ffffff !important;
        }
        
        /* 2. Style the main section (dropzone container) back to its subtle dark styling */
        [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
            background-color: rgba(255, 255, 255, 0.02) !important;
            border: 1.5px dashed rgba(99, 102, 241, 0.3) !important;
            border-radius: var(--radius-md) !important;
        }
        
        /* 3. Style the file cards specifically to have a dark translucent background with a clean border */
        [data-testid="stSidebar"] [data-testid="stUploadedFile"],
        [data-testid="stSidebar"] .uploadedFile,
        [data-testid="stSidebar"] [data-testid="stFileUploaderFile"],
        [data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stUploadedFile"],
        [data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
        [data-testid="stSidebar"] [data-testid="stFileUploader"] section [data-testid="stUploadedFile"],
        [data-testid="stSidebar"] [data-testid="stFileUploader"] section [data-testid="stFileUploaderFile"] {
            background-color: rgba(255, 255, 255, 0.08) !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            border-radius: var(--radius-md) !important;
            padding: 10px 14px !important;
            display: flex !important;
            align-items: center !important;
        }
        
        /* 4. Ensure all text and icons in the sidebar file cards are clean white */
        [data-testid="stSidebar"] [data-testid="stUploadedFile"] *,
        [data-testid="stSidebar"] .uploadedFile *,
        [data-testid="stSidebar"] [data-testid="stFileUploaderFile"] *,
        [data-testid="stSidebar"] [data-testid="stFileUploader"] section [data-testid="stUploadedFile"] *,
        [data-testid="stSidebar"] [data-testid="stFileUploader"] section [data-testid="stFileUploaderFile"] * {
            color: #ffffff !important;
            background-color: transparent !important;
        }
        
        [data-testid="stSidebar"] [data-testid="stUploadedFile"] svg,
        [data-testid="stSidebar"] .uploadedFile svg,
        [data-testid="stSidebar"] [data-testid="stFileUploaderFile"] svg,
        [data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stUploadedFile"] svg,
        [data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] svg,
        [data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] svg,
        [data-testid="stSidebar"] [data-testid="stFileUploader"] section [data-testid="stUploadedFile"] svg,
        [data-testid="stSidebar"] [data-testid="stFileUploader"] section [data-testid="stFileUploaderFile"] svg {
            fill: rgba(255, 255, 255, 0.8) !important;
            background-color: transparent !important;
        }
        
        /* 5. Style the add button (+) inside uploader section in the sidebar */
        [data-testid="stSidebar"] [data-testid="stFileUploader"] section button,
        [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
            background-color: rgba(255, 255, 255, 0.08) !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            color: #ffffff !important;
            border-radius: var(--radius-md) !important;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploader"] section button:hover,
        [data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover {
            background-color: rgba(255, 255, 255, 0.15) !important;
            border-color: rgba(255, 255, 255, 0.25) !important;
        }
        
        /* 6. Style the delete button (x) inside the card */
        [data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"],
        [data-testid="stSidebar"] [data-testid="stFileUploaderDeleteBtn"] {
            background-color: transparent !important;
            border: none !important;
            padding: 4px !important;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"]:hover,
        [data-testid="stSidebar"] [data-testid="stFileUploaderDeleteBtn"]:hover {
            background-color: rgba(255, 255, 255, 0.1) !important;
            border-radius: 50% !important;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            overflow: hidden;
        }

        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(15, 23, 42, 0.1);
            border-radius: 999px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(15, 23, 42, 0.2);
        }

        /* ═══ RESPONSIVE ═══ */
        @media (max-width: 900px) {
            .block-container {
                padding: 24px 20px 48px;
            }
            .studio-top, .studio-body {
                display: block;
                padding: 24px;
            }
            .studio h1 {
                font-size: 24px;
            }
            .studio-status {
                margin-top: 16px;
            }
            .workflow-card {
                margin-top: 16px;
                grid-template-columns: 1fr 1fr;
            }
            .kpi-grid {
                grid-template-columns: 1fr 1fr;
            }
            .empty-state, .document-header {
                display: block;
            }
            .glossary-grid, .prompt-grid {
                grid-template-columns: 1fr;
            }
            .doc-actions {
                justify-content: flex-start;
                margin-top: 16px;
            }
        }
        /* ═══ MODERN CARD-STYLE RADIO BUTTON LIST (ACTIVE STATE HIGHLIGHT) ═══ */
        /* Radio button vertical container layout */
        div[data-testid="stRadio"] > div {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        
        /* Default options style for light/main view */
        div[data-testid="stRadio"] label {
            background-color: #ffffff !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-md) !important;
            padding: 12px 18px !important;
            cursor: pointer !important;
            transition: var(--transition) !important;
            display: flex !important;
            align-items: center !important;
            gap: 12px !important;
            width: 100% !important;
            box-shadow: var(--shadow-sm) !important;
            color: var(--ink-secondary) !important;
        }
        
        div[data-testid="stRadio"] label:hover {
            background-color: var(--surface-hover) !important;
            border-color: var(--border-hover) !important;
        }
        
        /* Active highlight for selected radio option in main view */
        div[data-testid="stRadio"] label:has(input[checked]),
        div[data-testid="stRadio"] label:has(input:checked) {
            background-color: var(--primary-light) !important;
            border-color: var(--primary) !important;
            color: var(--primary) !important;
            font-weight: 700 !important;
            box-shadow: var(--shadow-md) !important;
        }

        /* Default options style for Sidebar view (Dark gradient backdrop) */
        [data-testid="stSidebar"] div[data-testid="stRadio"] label {
            background-color: rgba(255, 255, 255, 0.03) !important;
            border: 1px solid rgba(255, 255, 255, 0.06) !important;
            border-radius: var(--radius-md) !important;
            color: rgba(241, 245, 249, 0.85) !important;
            box-shadow: none !important;
        }

        [data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
            background-color: rgba(255, 255, 255, 0.08) !important;
            border-color: rgba(255, 255, 255, 0.15) !important;
            color: #ffffff !important;
        }

        /* Active highlight for selected radio option in Sidebar view */
        [data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input[checked]),
        [data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {
            background-color: rgba(99, 102, 241, 0.15) !important;
            border-color: rgba(99, 102, 241, 0.45) !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            box-shadow: 0 4px 14px rgba(99, 102, 241, 0.15) !important;
        }
                /* EXPANDER / KAYNAK PARÇALAR STYLING */
        [data-testid="stExpander"] {
            background-color: var(--card-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 12px !important;
            box-shadow: var(--shadow-sm) !important;
            overflow: hidden !important;
            margin-top: 1rem !important;
        }
        [data-testid="stExpander"] details summary {
            background-color: var(--bg-color) !important;
            color: var(--text-color) !important;
            font-weight: 500 !important;
            padding: 12px 16px !important;
            border-bottom: 1px solid var(--border-color) !important;
        }
        [data-testid="stExpander"] details summary:hover {
            background-color: var(--hover-color) !important;
            color: var(--text-primary) !important;
        }
        [data-testid="stExpander"] details summary svg {
            fill: var(--text-color) !important;
        }
        [data-testid="stExpander"] .st-emotion-cache-16idsys p {
            color: var(--text-secondary) !important;
            font-size: 0.95rem !important;
        }
        
                /* QUIZ RADIO BUTTONS STYLING */
        [data-testid="stRadio"] > div {
            gap: 0.5rem !important;
        }
        [data-testid="stRadio"] label {
            background-color: var(--bg-color) !important;
            padding: 12px 16px !important;
            border-radius: 8px !important;
            border: 1px solid var(--border-color) !important;
            transition: all 0.2s ease !important;
            cursor: pointer !important;
            width: 100% !important;
            display: flex !important;
            align-items: center !important;
        }
        [data-testid="stRadio"] label:hover {
            border-color: var(--primary-color) !important;
            background-color: var(--hover-color) !important;
        }
        [data-testid="stRadio"] label[data-baseweb="radio"] div:first-child {
            margin-right: 12px !important;
        }
        
                /* AI BUBBLE AND CHAT INPUT STYLING */
        .ai-bubble {
            display: flex;
            align-items: center;
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            padding: 1rem 1.5rem;
            border-radius: 20px 20px 20px 4px;
            color: white;
            box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.4);
            margin-bottom: 2rem;
            animation: float 3s ease-in-out infinite;
            gap: 1rem;
        }
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-5px); }
            100% { transform: translateY(0px); }
        }
        .ai-bubble-icon {
            font-size: 2rem;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
        .ai-bubble-content strong {
            display: block;
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
            color: white;
        }
        .ai-bubble-content span {
            font-size: 0.95rem;
            opacity: 0.95;
            line-height: 1.4;
            color: white;
        }
        
        /* Modern Chat Input Styling override */
        div[data-testid="stChatInput"] {
            border-radius: 12px !important;
            border: 1px solid var(--border-color) !important;
            background-color: var(--card-bg) !important;
            box-shadow: var(--shadow-sm) !important;
            transition: all 0.2s ease;
        }
        div[data-testid="stChatInput"]:focus-within {
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 2px rgba(99,102,241,0.2) !important;
        }
        
                /* GLOSSARY CARD STYLING */
        .glossary-card {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.25rem;
            height: 100%;
            min-height: 140px;
            box-shadow: var(--shadow-sm);
            transition: all 0.3s ease;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            position: relative;
            overflow: hidden;
        }
        .glossary-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: linear-gradient(to bottom, #6366f1, #a855f7);
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        .glossary-card:hover {
            transform: translateY(-4px);
            box-shadow: var(--shadow-md);
            border-color: rgba(99, 102, 241, 0.3);
        }
        .glossary-card:hover::before {
            opacity: 1;
        }
        .glossary-term {
            font-weight: 700;
            font-size: 1.05rem;
            color: var(--primary-color);
            line-height: 1.3;
        }
        .glossary-desc {
            font-size: 0.95rem;
            color: var(--text-secondary);
            line-height: 1.5;
        }

                /* PREMIUM SUMMARY CARD STYLING */
        .premium-summary-card {
            background-color: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 2.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            margin-top: 1.5rem;
            color: var(--ink-secondary);
        }
        .premium-summary-card h1 {
            color: var(--primary);
            font-size: 2.2rem;
            margin-bottom: 1.2rem;
            border-bottom: 2px solid var(--primary-light);
            padding-bottom: 0.8rem;
            font-weight: 800;
        }
        .premium-summary-card h2 {
            color: var(--ink-primary);
            font-size: 1.6rem;
            margin-top: 2.5rem;
            margin-bottom: 1rem;
            font-weight: 700;
        }
        .premium-summary-card p {
            line-height: 1.8;
            font-size: 1.05rem;
            color: var(--ink-secondary);
            margin-bottom: 1.2rem;
        }
        .premium-summary-card ul {
            padding-left: 1.5rem;
            margin-bottom: 1.5rem;
        }
        .premium-summary-card li {
            margin-bottom: 0.6rem;
            line-height: 1.7;
            color: var(--ink-secondary);
        }
        .premium-summary-card strong {
            color: var(--ink-primary);
            font-weight: 700;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )





if __name__ == "__main__":
    main()

