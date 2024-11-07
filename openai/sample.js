const cells = document.querySelectorAll(".cell");
const statusText = document.querySelector("#statusText");
const restartBtn = document.querySelector("#restartBtn");
const winConditions = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6]
];
let options = ["", "", "", "", "", "", "", "", ""];
let currentPlayer = document.createElement("p")
currentPlayer.innerText = "tomato"
let running = false;


const sn0 = document.getElementById("sn0")
const sn1 = document.getElementById("sn1")
const sn2 = document.getElementById("sn2")
const sn3 = document.getElementById("sn3")
const sn4 = document.getElementById("sn4")
const sn5 = document.getElementById("sn5")
const sn6 = document.getElementById("sn6")
const sn7 = document.getElementById("sn7")
const sn8 = document.getElementById("sn8")


initializeGame();

function initializeGame(){
    cells.forEach(cell => cell.addEventListener("click", cellClicked));
    restartBtn.addEventListener("click", restartGame);
    statusText.textContent = `${currentPlayer}'s turn`;
    running = true;
}
function cellClicked(){
    const cellIndex = this.getAttribute("cellIndex");

    if(options[cellIndex] != "" || !running){
        return;
    }

    updateCell(this, cellIndex);
    checkWinner();
}

function updateCell(cell, index){
    options[index] = currentPlayer.innerText;
    cell.textContent = currentPlayer.innerText;
}
function changePlayer(){
    currentPlayer.innerText = (currentPlayer.innerText == "tomato") ? "flower" : "tomato";
    statusText.textContent = `${currentPlayer.innerText}'s turn`;

    
}

function checkWinner(){
    let roundWon = false;

    for(let i = 0; i < winConditions.length; i++){
        const condition = winConditions[i];
        const cellA = options[condition[0]];
        const cellB = options[condition[1]];
        const cellC = options[condition[2]];

        if(cellA == "" || cellB == "" || cellC == ""){
            continue;
        }
        if(cellA == cellB && cellB == cellC){
            roundWon = true;
            break;
        }
    }

    if(roundWon){
        statusText.textContent = `${currentPlayer.innerText} wins!`;
        running = false;
    }
    else if(!options.includes("")){
        statusText.textContent = `Draw!`;
        running = false;
    }
    else{
        changePlayer();
    }

    if (sn0.innerHTML.includes("tomato")) {
        sn0.classList.add("tomato")
    } else if (sn0.innerHTML.includes("flower")){
        sn0.classList.add("flower")
    }

    if (sn1.innerHTML.includes("tomato")) {
        sn1.classList.add("tomato")
    } else if (sn1.innerHTML.includes("flower")){
        sn1.classList.add("flower")
    }

    if (sn2.innerHTML.includes("tomato")) {
        sn2.classList.add("tomato")
    } else if (sn2.innerHTML.includes("flower")){
        sn2.classList.add("flower")
    }

    if (sn3.innerHTML.includes("tomato")) {
        sn3.classList.add("tomato")
    } else if (sn3.innerHTML.includes("flower")){
        sn3.classList.add("flower")
    }

    if (sn4.innerHTML.includes("tomato")) {
        sn4.classList.add("tomato")
    } else if (sn4.innerHTML.includes("flower")){
        sn4.classList.add("flower")
    }

    if (sn5.innerHTML.includes("tomato")) {
        sn5.classList.add("tomato")
    } else if (sn5.innerHTML.includes("flower")){
        sn5.classList.add("flower")
    }

    if (sn6.innerHTML.includes("tomato")) {
        sn6.classList.add("tomato")
    } else if (sn6.innerHTML.includes("flower")){
        sn6.classList.add("flower")
    }

    if (sn7.innerHTML.includes("tomato")) {
        sn7.classList.add("tomato")
    } else if (sn7.innerHTML.includes("flower")){
        sn7.classList.add("flower")
    }

    if (sn8.innerHTML.includes("tomato")) {
        sn8.classList.add("tomato")
    } else if (sn8.innerHTML.includes("flower")){
        sn8.classList.add("flower")
    }
    




}

function restartGame(){
    currentPlayer.innerText = "tomato";
    options = ["", "", "", "", "", "", "", "", ""];
    statusText.textContent = `${currentPlayer.innerText}'s turn`;
    cells.forEach(cell => cell.textContent = "");
    running = true;
    
    sn0.classList.remove("tomato")
    sn0.classList.remove("flower")

    sn1.classList.remove("tomato")
    sn1.classList.remove("flower")

    sn2.classList.remove("tomato")
    sn2.classList.remove("flower")

    sn3.classList.remove("tomato")
    sn3.classList.remove("flower")

    sn4.classList.remove("tomato")
    sn4.classList.remove("flower")

    sn5.classList.remove("tomato")
    sn5.classList.remove("flower")

    