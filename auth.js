// firebase-auth.js

// Import Firebase modules
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-app.js";
import { 
    getAuth, 
    createUserWithEmailAndPassword, 
    signInWithEmailAndPassword,
    signInWithPopup,
    GoogleAuthProvider,
    signOut,
    onAuthStateChanged
} from "https://www.gstatic.com/firebasejs/10.7.1/firebase-auth.js";

// Firebase configuration
const firebaseConfig = {
    apiKey: "AIzaSyBn6eWdMPx97cocWhjIgQjpaVC2SNhCmI0",
    authDomain: "abhinav-3416f.firebaseapp.com",
    projectId: "abhinav-3416f",
    storageBucket: "abhinav-3416f.firebasestorage.app",
    messagingSenderId: "1088697904304",
    appId: "1:1088697904304:web:0fd1b529eb59bd30bd5f75"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const googleProvider = new GoogleAuthProvider();

// Get DOM elements
const emailInput = document.querySelector('input[type="email"]');
const passwordInput = document.querySelector('input[type="password"]');
const signInBtn = document.querySelector('.sign-in-btn');
const googleSignInBtn = document.querySelector('.google-sign-in-btn');
const createAccountLink = document.querySelector('.create-account-link');

// Sign In with Email and Password
async function signInWithEmail(email, password) {
    try {
        const userCredential = await signInWithEmailAndPassword(auth, email, password);
        const user = userCredential.user;
        
        showMessage('Sign in successful!', 'success');
        console.log('Signed in user:', user);
        
        // Redirect to dashboard after successful login
        setTimeout(() => {
            window.location.href = 'index.html'; // Change to your dashboard page
        }, 1000);
        
    } catch (error) {
        handleAuthError(error);
    }
}

// Sign Up with Email and Password
async function signUpWithEmail(email, password) {
    try {
        const userCredential = await createUserWithEmailAndPassword(auth, email, password);
        const user = userCredential.user;
        
        showMessage('Account created successfully!', 'success');
        console.log('Created user:', user);
        
        // Redirect to dashboard
        setTimeout(() => {
            window.location.href = 'index.html';
        }, 1000);
        
    } catch (error) {
        handleAuthError(error);
    }
}

// Sign In with Google
async function signInWithGoogle() {
    try {
        const result = await signInWithPopup(auth, googleProvider);
        const user = result.user;
        
        showMessage('Google sign in successful!', 'success');
        console.log('Google user:', user);
        
        // Redirect to dashboard
        setTimeout(() => {
            window.location.href = 'index.html';
        }, 1000);
        
    } catch (error) {
        handleAuthError(error);
    }
}

// Sign Out
async function signOutUser() {
    try {
        await signOut(auth);
        showMessage('Signed out successfully', 'success');
        window.location.href = 'login.html';
    } catch (error) {
        showMessage('Error signing out: ' + error.message, 'error');
    }
}

// Handle Authentication Errors
function handleAuthError(error) {
    const errorCode = error.code;
    const errorMessage = error.message;
    
    console.error('Auth error:', errorCode, errorMessage);
    
    let userMessage = '';
    
    switch(errorCode) {
        case 'auth/invalid-email':
            userMessage = 'Invalid email address format.';
            break;
        case 'auth/user-disabled':
            userMessage = 'This account has been disabled.';
            break;
        case 'auth/user-not-found':
            userMessage = 'No account found with this email.';
            break;
        case 'auth/wrong-password':
            userMessage = 'Incorrect password.';
            break;
        case 'auth/email-already-in-use':
            userMessage = 'Email already in use. Please login instead.';
            break;
        case 'auth/weak-password':
            userMessage = 'Password should be at least 6 characters.';
            break;
        case 'auth/popup-closed-by-user':
            userMessage = 'Sign in popup was closed.';
            break;
        case 'auth/invalid-credential':
            userMessage = 'Invalid email or password.';
            break;
        default:
            userMessage = 'Authentication error: ' + errorMessage;
    }
    
    showMessage(userMessage, 'error');
}

// Show Message to User
function showMessage(message, type) {
    // Remove existing messages
    const existingMsg = document.querySelector('.auth-message');
    if (existingMsg) {
        existingMsg.remove();
    }
    
    // Create message element
    const messageDiv = document.createElement('div');
    messageDiv.className = `auth-message ${type}`;
    messageDiv.textContent = message;
    messageDiv.style.cssText = `
        padding: 12px 20px;
        margin: 15px 0;
        border-radius: 8px;
        text-align: center;
        font-size: 14px;
        animation: slideIn 0.3s ease;
        ${type === 'success' ? 'background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb;' : 
          'background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb;'}
    `;
    
    // Insert before sign in button
    signInBtn.parentElement.insertBefore(messageDiv, signInBtn);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        messageDiv.remove();
    }, 5000);
}

// Validate Email Format
function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

// Toggle between Login and Signup modes
let isSignUpMode = false;

function toggleMode() {
    isSignUpMode = !isSignUpMode;
    
    if (isSignUpMode) {
        signInBtn.textContent = 'Sign Up';
        createAccountLink.innerHTML = 'Already have an account? <strong>Sign In</strong>';
    } else {
        signInBtn.textContent = 'Sign In';
        createAccountLink.innerHTML = 'Don\'t have an account? <strong>Create Account</strong>';
    }
}

// Event Listeners
if (signInBtn) {
    signInBtn.addEventListener('click', async (e) => {
        e.preventDefault();
        
        const email = emailInput.value.trim();
        const password = passwordInput.value.trim();
        
        // Validation
        if (!email || !password) {
            showMessage('Please fill in all fields', 'error');
            return;
        }
        
        if (!isValidEmail(email)) {
            showMessage('Please enter a valid email address', 'error');
            return;
        }
        
        if (password.length < 6) {
            showMessage('Password must be at least 6 characters', 'error');
            return;
        }
        
        // Disable button during processing
        signInBtn.disabled = true;
        signInBtn.textContent = 'Processing...';
        
        // Call appropriate function
        if (isSignUpMode) {
            await signUpWithEmail(email, password);
        } else {
            await signInWithEmail(email, password);
        }
        
        // Re-enable button
        signInBtn.disabled = false;
        signInBtn.textContent = isSignUpMode ? 'Sign Up' : 'Sign In';
    });
}

if (googleSignInBtn) {
    googleSignInBtn.addEventListener('click', async (e) => {
        e.preventDefault();
        
        googleSignInBtn.disabled = true;
        const originalText = googleSignInBtn.textContent;
        googleSignInBtn.textContent = 'Processing...';
        
        await signInWithGoogle();
        
        googleSignInBtn.disabled = false;
        googleSignInBtn.textContent = originalText;
    });
}

if (createAccountLink) {
    createAccountLink.addEventListener('click', (e) => {
        e.preventDefault();
        toggleMode();
    });
}

// Monitor Authentication State
onAuthStateChanged(auth, (user) => {
    if (user) {
        console.log('User is signed in:', user);
        
        // If on login page and user is already logged in, redirect to dashboard
        if (window.location.pathname.includes('login.html')) {
            window.location.href = 'index.html';
        }
    } else {
        console.log('No user signed in');
        
        // If on protected page and no user, redirect to login
        if (!window.location.pathname.includes('login.html')) {
            // Uncomment if you want to protect other pages
            // window.location.href = 'login.html';
        }
    }
});

// Export functions for use in other files
export { auth, signOutUser, onAuthStateChanged };
