document.addEventListener('DOMContentLoaded', function() {
    const container = document.getElementById('container');
    const registerBtn = document.getElementById('register');
    const loginBtn = document.getElementById('login');
    const servicesBtn = document.querySelector('.services-btn');
    const contactBtn = document.querySelector('.contact-btn')
    const signUpForm = document.querySelector('.sign-up form');
    const signInForm = document.querySelector('.sign-in form');
    const emailInput = signInForm.querySelector('input[type="email"]');
    const passwordInput = signInForm.querySelector('input[type="password"]');

    // Function to toggle to the registration form
    function activateRegister() {
        if (container) container.classList.add("active");
    }

    // Function to toggle to the login form
    function activateLogin() {
        if (container) container.classList.remove("active");
    }

    // Setup event listeners if elements exist
    if (registerBtn) {
        registerBtn.addEventListener('click', activateRegister);
    } else {
        console.log("Register button not found.");
    }

    if (loginBtn) {
        loginBtn.addEventListener('click', activateLogin);
    } else {
        console.log("Login button not found.");
    }

    if (contactBtn) {
        contactBtn.addEventListener('click', function() {
            window.location.href = 'contact.html';
        });
    } else {
        console.log("Services button not found.");
    }

    // Services button event listener for navigation
    if (servicesBtn) {
        servicesBtn.addEventListener('click', function() {
            window.location.href = 'services.html';
        });
    } else {
        console.log("Services button not found.");
    }

    // Handle the sign-up form submission
    if (signUpForm) {
        signUpForm.addEventListener('submit', function(event) {
            event.preventDefault(); // Prevent the default form submission

            const email = signUpForm.querySelector('input[type="email"]').value;
            const password = signUpForm.querySelector('input[type="password"]').value;

            // Save credentials in localStorage
            localStorage.setItem('email', email);
            localStorage.setItem('password', password);

            // After signing up, automatically switch to the login form
            activateLogin();
        });
    } else {
        console.log("Sign-up form not found.");
    }

    // Handle the sign-in form submission
    if (signInForm) {
        signInForm.addEventListener('submit', function(event) {
            event.preventDefault(); // Prevent the default form submission

            const storedEmail = localStorage.getItem('email');
            const storedPassword = localStorage.getItem('password');

            const email = emailInput.value;
            const password = passwordInput.value;

            if (email === storedEmail && password === storedPassword) {
                // Redirect to the Admin Dashboard page
                window.location.href = 'AdminDashboard/index.html';
            } else {
                alert('Incorrect email or password, please try again.');
            }
        });
    } else {
        console.log("Sign-in form not found.");
    }
});
