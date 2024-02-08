const container = document.getElementById('container');
const swap2btn = document.getElementById('swap2');
const swap1btn = document.getElementById('swap1');
const registerBtn = document.getElementById('resgister');
const loginBtn = document.getElementById('signin');

swap2btn.addEventListener('click', () => {
    container.classList.add("active");
});

swap1btn.addEventListener('click', () => {
    container.classList.remove("active");
});

window.addEventListener('DOMContentLoaded', (event) => {
    const signinBtn = document.getElementById('Signin');
    signinBtn.addEventListener('click', (e) => {
        e.preventDefault();

        const form = document.getElementById('sign');
        const email = form.querySelector('input[type="email"]').value;
        const password = form.querySelector('input[type="password"]').value;

        fetch('http://localhost:8000', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ email: email, password: password }),
        })
        .then(response => response.text())
        .then(data => console.log(data))
        .catch(error => console.error('Error:', error));
    });
});