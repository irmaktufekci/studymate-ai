 StudyMate AI

StudyMate AI, akademik PDF ve TXT dosyalarını analiz eden Streamlit tabanlı bir yapay zeka çalışma asistanıdır. Kullanıcıların yüklediği belgeler üzerinden özet çıkarma, kavram analizi ve soru-cevap işlemleri yapar.

 Özellikler
PDF ve TXT dosya yükleme
PDF metin çıkarma (pypdf)
Metin parçalama (LangChain RecursiveCharacterTextSplitter)
HuggingFace sentence-transformers ile embedding
FAISS ile yerel vektör veritabanı
RAG tabanlı soru-cevap sistemi
Otomatik özet çıkarma
Kavram sözlüğü oluşturma
OpenAI / Groq API desteği (opsiyonel)
API yoksa fallback (extractive) çalışma modu
Dosya ve indeks yönetimi
⚙️ Kurulum
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env

 Ortam Değişkenleri (.env)
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

GROQ_API_KEY=gsk-...
GROQ_MODEL=llama-3.3-70b-versatile

Not: Groq kullanırsan OpenAI zorunlu değildir. API key girmezsen sistem fallback modda çalışır.

 Çalıştırma
streamlit run app.py

Uygulama varsayılan olarak şu adreste açılır:

http://localhost:8501
 Sistem Yapısı
StudyMate AI
├─ app.py
├─ studymate/
│  ├─ config.py
│  ├─ document_loader.py
│  ├─ rag.py
│  └─ storage.py
└─ data/
   ├─ uploads/
   └─ indexes/
   
 Geliştirme Süreci
Proje iskeleti oluşturuldu
PDF/TXT yükleme ve metin çıkarma eklendi
Chunking ve embedding sistemi kuruldu
FAISS vektör veritabanı entegre edildi
RAG tabanlı soru-cevap sistemi geliştirildi
Özet ve kavram sözlüğü modülleri eklendi
Streamlit arayüzü tamamlandı

 Notlar
İlk çalıştırmada embedding modeli internetten indirilebilir
OpenAI API olmadan da sistem çalışır (fallback mod)
FAISS ile yerel arama yapılır
Groq ve OpenAI opsiyoneldir
📌 Amaç

Bu proje, akademik dokümanları daha hızlı anlamayı ve etkileşimli şekilde öğrenmeyi amaçlayan bir AI destekli çalışma asistanıdır.
