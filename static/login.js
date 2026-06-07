document.addEventListener('DOMContentLoaded', () => {
    const sendOtpBtn = document.getElementById('sendOtpBtn');
    const loginForm = document.getElementById('loginForm');
    const step1 = document.getElementById('step1');
    const step2 = document.getElementById('step2');
    const emailInput = document.getElementById('email');
    const passwordInput = document.getElementById('password');
    const otpInput = document.getElementById('otp');
    const loginMessage = document.getElementById('loginMessage');
    const loginBtn = document.getElementById('loginBtn');

    // Tab items
    const tabSignIn = document.getElementById('tabSignIn');
    const tabSignUp = document.getElementById('tabSignUp');
    const nameField = document.getElementById('nameField');
    const fullnameInput = document.getElementById('fullname');
    const heading = document.getElementById('heading');

    // Timer elements
    const otpTimer = document.getElementById('otpTimer');
    const otpTimerContainer = document.getElementById('otpTimerContainer');
    const resendOtpContainer = document.getElementById('resendOtpContainer');
    const resendOtpBtn = document.getElementById('resendOtpBtn');

    let countdown;
    let timeLeft = 30;
    let currentMode = 'signin'; // 'signin' or 'signup'

    // -----------------------------------------------------------------------
    // MODE SWITCHING (Sign In vs Sign Up Tabs)
    // -----------------------------------------------------------------------
    tabSignIn.addEventListener('click', () => {
        if (currentMode === 'signin') return;
        currentMode = 'signin';
        
        tabSignIn.classList.add('active');
        tabSignUp.classList.remove('active');
        nameField.classList.add('hidden');
        fullnameInput.required = false;
        
        heading.innerText = 'Sign In to Your Account';
        sendOtpBtn.innerText = 'Send Verification OTP';
        loginBtn.innerText = 'Verify & Login';
        loginMessage.innerText = '';
    });

    tabSignUp.addEventListener('click', () => {
        if (currentMode === 'signup') return;
        currentMode = 'signup';
        
        tabSignUp.classList.add('active');
        tabSignIn.classList.remove('active');
        nameField.classList.remove('hidden');
        fullnameInput.required = true;
        
        heading.innerText = 'Create a New Account';
        sendOtpBtn.innerText = 'Send Registration OTP';
        loginBtn.innerText = 'Verify & Register';
        loginMessage.innerText = '';
    });

    // -----------------------------------------------------------------------
    // OTP TIMER CONTROL
    // -----------------------------------------------------------------------
    function startOtpTimer() {
        clearInterval(countdown);
        timeLeft = 30;
        
        otpInput.disabled = false;
        loginBtn.disabled = false;
        otpTimerContainer.classList.remove('hidden');
        resendOtpContainer.classList.add('hidden');
        otpTimer.innerText = timeLeft;
        otpInput.value = ''; 

        countdown = setInterval(() => {
            timeLeft--;
            otpTimer.innerText = timeLeft;
            if (timeLeft <= 0) {
                clearInterval(countdown);
                otpInput.disabled = true;
                loginBtn.disabled = true;
                otpTimerContainer.classList.add('hidden');
                resendOtpContainer.classList.remove('hidden');
                showMessage("OTP expired. Please generate a new one.", "error-text");
            }
        }, 1000);
    }

    // -----------------------------------------------------------------------
    // REQUEST OTP
    // -----------------------------------------------------------------------
    sendOtpBtn.addEventListener('click', () => {
        const email = emailInput.value.trim();
        const password = passwordInput.value;
        const fullname = fullnameInput.value.trim();

        if (currentMode === 'signup' && !fullname) {
            showMessage("Please enter your full name.", "error-text");
            return;
        }

        if (!email || !password) {
            showMessage("Please enter both email and password.", "error-text");
            return;
        }

        const originalText = sendOtpBtn.innerText;
        sendOtpBtn.innerText = "Sending...";
        sendOtpBtn.disabled = true;

        fetch('/send-otp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: email })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                step1.classList.add('hidden');
                step2.classList.remove('hidden');
                const successMsg = currentMode === 'signup' 
                    ? "Registration OTP sent! Check your inbox." 
                    : "Verification OTP sent! Check your inbox.";
                showMessage(successMsg, "success-text");
                startOtpTimer();
            } else {
                sendOtpBtn.innerText = originalText;
                sendOtpBtn.disabled = false;
                showMessage(data.message || "Failed to send OTP", "error-text");
            }
        })
        .catch(err => {
            sendOtpBtn.innerText = originalText;
            sendOtpBtn.disabled = false;
            showMessage("Server error while sending OTP.", "error-text");
        });
    });

    // -----------------------------------------------------------------------
    // RESEND OTP
    // -----------------------------------------------------------------------
    resendOtpBtn.addEventListener('click', () => {
        const email = emailInput.value.trim();

        resendOtpBtn.innerText = "Generating...";
        resendOtpBtn.disabled = true;

        fetch('/send-otp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: email })
        })
        .then(response => response.json())
        .then(data => {
            resendOtpBtn.innerText = "Resend / Generate New OTP";
            resendOtpBtn.disabled = false;
            if (data.status === 'success') {
                showMessage("A new OTP has been sent successfully!", "success-text");
                startOtpTimer();
            } else {
                showMessage(data.message || "Failed to resend OTP", "error-text");
            }
        })
        .catch(err => {
            resendOtpBtn.innerText = "Resend / Generate New OTP";
            resendOtpBtn.disabled = false;
            showMessage("Server error while resending OTP.", "error-text");
        });
    });

    // -----------------------------------------------------------------------
    // VERIFY & SUBMIT FORM
    // -----------------------------------------------------------------------
    loginForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const email = emailInput.value.trim();
        const password = passwordInput.value;
        const otp = otpInput.value.trim();

        if (!otp) {
            showMessage("Please enter the OTP.", "error-text");
            return;
        }

        const originalBtnText = loginBtn.innerText;
        loginBtn.innerText = "Verifying...";
        loginBtn.disabled = true;

        fetch('/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: email, password: password, otp: otp })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                clearInterval(countdown);
                const successMsg = currentMode === 'signup' 
                    ? "Registration Successful! Welcome to RoadVision." 
                    : "Login Successful! Redirecting...";
                showMessage(successMsg, "success-text");
                window.location.href = "/";
            } else {
                loginBtn.innerText = originalBtnText;
                loginBtn.disabled = false;
                showMessage(data.message || "Invalid OTP or Credentials.", "error-text");
            }
        })
        .catch(err => {
            loginBtn.innerText = originalBtnText;
            loginBtn.disabled = false;
            showMessage("Server error while logging in.", "error-text");
        });
    });

    function showMessage(msg, className) {
        loginMessage.innerText = msg;
        loginMessage.className = "message-text " + className;
    }
});
