const hindiChars = ["अ", "आ", "इ", "ई", "उ", "ऊ", "ए", "ऐ", "ओ", "औ", "क", "ख", "ग", "घ", "च", "छ", "ज", "झ", "ट", "ठ", "ड", "ढ", "त", "थ", "द", "ध", "न", "प", "फ", "ब", "भ", "म", "य", "र", "ल", "व", "श", "ष", "स", "ह"];

function createHindiChar() {
    const char = document.createElement("div");
    char.classList.add("hindi-char");
    char.innerText = hindiChars[Math.floor(Math.random() * hindiChars.length)];
    char.style.left = Math.random() * 100 + "vw";
    char.style.animationDuration = 8 + Math.random() * 6 + "s"; 
    char.style.fontSize = (20 + Math.random() * 40) + "px";
    document.getElementById("floating-background").appendChild(char);

    setTimeout(() => {
        char.remove();
    }, 12000);
}

setInterval(createHindiChar, 200);
