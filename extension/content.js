const siteSelectors = {
  "chat.openai.com": "textarea",
  "claude.ai": 'div[contenteditable="true"]',
  "gemini.google.com": "textarea",
  "perplexity.ai": 'textarea[placeholder*=\"Ask anything\"]'
};

function getSelectorForSite() {
  const host = window.location.hostname;
  return siteSelectors[host] || null;
}

function createButton(textarea) {
  if (document.getElementById("enhance-btn")) return; // avoid duplicates

  const btn = document.createElement("button");
  btn.id = "enhance-btn";
  btn.innerText = "Enhance Prompt";
  btn.style.margin = "5px";
  btn.style.padding = "5px 10px";
  btn.style.background = "#4CAF50";
  btn.style.color = "white";
  btn.style.border = "none";
  btn.style.borderRadius = "5px";
  btn.style.cursor = "pointer";

  textarea.parentNode.appendChild(btn);

  btn.addEventListener("click", async () => {
    let prompt = textarea.value || textarea.innerText;
    
    let response = await fetch("http://localhost:5000/enhance", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt })
    });

    let data = await response.json();
    showModal(textarea, data.original, data.enhanced);
  });
}

function showModal(textarea, original, enhanced) {
  // remove existing modal if any
  const oldModal = document.getElementById("enhance-modal");
  if (oldModal) oldModal.remove();

  const modal = document.createElement("div");
  modal.id = "enhance-modal";
  modal.style.position = "fixed";
  modal.style.top = "20%";
  modal.style.left = "50%";
  modal.style.transform = "translate(-50%, -20%)";
  modal.style.background = "white";
  modal.style.padding = "20px";
  modal.style.border = "1px solid #ccc";
  modal.style.borderRadius = "10px";
  modal.style.boxShadow = "0 4px 8px rgba(0,0,0,0.2)";
  modal.style.zIndex = "9999";

  modal.innerHTML = `
    <h3>Choose Prompt</h3>
    <p><b>Original:</b> ${original}</p>
    <p><b>Enhanced:</b> ${enhanced}</p>
    <button id="use-original">Use Original</button>
    <button id="use-enhanced">Use Enhanced</button>
  `;

  document.body.appendChild(modal);

  document.getElementById("use-original").onclick = () => {
    textarea.value = original;
    textarea.innerText = original;
    modal.remove();
  };

  document.getElementById("use-enhanced").onclick = () => {
    textarea.value = enhanced;
    textarea.innerText = enhanced;
    modal.remove();
  };
}

function init() {
  const selector = getSelectorForSite();
  if (!selector) return;

  const textarea = document.querySelector(selector);
  if (textarea) {
    createButton(textarea);
  } else {
    // watch for dynamic loading
    const observer = new MutationObserver(() => {
      const ta = document.querySelector(selector);
      if (ta) {
        createButton(ta);
        observer.disconnect();
      }
    });
    observer.observe(document.body, { childList: true, subtree: true });
  }
}

init();
