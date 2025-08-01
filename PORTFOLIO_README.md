# AI & Machine Learning Portfolio 🤖

A modern, responsive portfolio website showcasing AI and Machine Learning projects with beautiful animations and interactive features.

![Portfolio Demo](https://img.shields.io/badge/Status-Live-brightgreen)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black)
![Responsive](https://img.shields.io/badge/Responsive-Design-blue)

## ✨ Features

### 🎨 Modern Design
- **Responsive Layout**: Works perfectly on all devices (desktop, tablet, mobile)
- **Modern UI/UX**: Clean, professional design with smooth animations
- **Custom Animations**: Fade-in effects, hover animations, and scroll-triggered animations
- **Gradient Backgrounds**: Beautiful gradient color schemes throughout

### 🚀 Interactive Elements
- **Smooth Scrolling**: Seamless navigation between sections
- **Mobile-First Navigation**: Responsive hamburger menu for mobile devices
- **Animated Skill Bars**: Progressive skill visualization
- **Counter Animations**: Animated statistics counters
- **Button Ripple Effects**: Material Design-inspired button interactions

### 📱 Mobile Responsive
- **Breakpoints**: Optimized for all screen sizes
- **Touch-Friendly**: Easy navigation on touch devices
- **Fast Loading**: Optimized performance and loading times

### 📊 Project Showcase
- **Featured Project**: Highlighted AI Attendance System project
- **Tech Stack Display**: Clear technology badges for each project
- **Project Features**: Detailed feature lists with checkmarks
- **Call-to-Action Buttons**: Links to demos and source code

### 📬 Contact Integration
- **Contact Form**: Functional contact form with validation
- **Social Links**: Direct links to social media profiles
- **Notification System**: User feedback for form submissions

## 🛠️ Technologies Used

- **HTML5**: Semantic markup and modern web standards
- **CSS3**: Custom properties, Flexbox, Grid, animations
- **JavaScript (ES6+)**: Modern JavaScript features and APIs
- **Font Awesome**: Icon library for consistent iconography
- **Google Fonts**: Inter font family for clean typography

## 🚀 Quick Start

### Prerequisites
- Web browser (Chrome, Firefox, Safari, Edge)
- Python 3.x (for local development server) OR Node.js (for live-server)

### Installation

1. **Clone or Download**
   ```bash
   git clone https://github.com/yourusername/ai-ml-portfolio.git
   cd ai-ml-portfolio
   ```

2. **Using Python (Recommended)**
   ```bash
   # Python 3
   python -m http.server 8000
   
   # Python 2 (if needed)
   python -m SimpleHTTPServer 8000
   ```

3. **Using Node.js (Alternative)**
   ```bash
   # Install live-server globally
   npm install -g live-server
   
   # Or use npx (no installation needed)
   npx live-server --port=8000
   ```

4. **Using npm scripts**
   ```bash
   # Install dependencies
   npm install
   
   # Start development server
   npm start
   
   # Or use live-server
   npm run live-server
   ```

5. **Open in Browser**
   Navigate to `http://localhost:8000` in your web browser

## 📁 Project Structure

```
ai-ml-portfolio/
├── index.html              # Main HTML file
├── style.css               # CSS styles and animations
├── script.js               # JavaScript functionality
├── package.json            # Project configuration
├── PORTFOLIO_README.md     # This documentation
├── README.md               # Original AI project documentation
├── AI Project Files/
│   ├── 1_datasetCreation.py
│   ├── 2_preprocessingEmbeddings.py
│   ├── 3_trainingFaceML.py
│   ├── 5_recognizingPersonwithCSVDatabase.py
│   ├── haarcascade_frontalface_default.xml
│   ├── openface-nn4.small2.v1.t7
│   ├── student.csv
│   ├── model/
│   └── output/
```

## 🎨 Customization Guide

### 1. Personal Information
Update the following sections in `index.html`:

```html
<!-- Hero Section -->
<h1 class="hero-title">
    Hi, I'm [Your Name] <span class="highlight">AI & ML Developer</span>
</h1>

<!-- About Section -->
<p>Your personal description...</p>

<!-- Contact Information -->
<div class="contact-method">
    <i class="fas fa-envelope"></i>
    <span>your.email@example.com</span>
</div>
```

### 2. Colors and Branding
Modify CSS custom properties in `style.css`:

```css
:root {
    --primary-color: #6366f1;      /* Main brand color */
    --secondary-color: #ec4899;     /* Accent color */
    --accent-color: #06b6d4;       /* Highlight color */
    /* Adjust other colors as needed */
}
```

### 3. Projects Section
Add your own projects by modifying the projects grid in `index.html`:

```html
<div class="project-card">
    <div class="project-image">
        <div class="project-icon">
            <i class="fas fa-your-icon"></i>
        </div>
    </div>
    <div class="project-content">
        <h3>Your Project Name</h3>
        <p>Project description...</p>
        <div class="project-tech">
            <span class="tech-tag">Technology 1</span>
            <span class="tech-tag">Technology 2</span>
        </div>
        <div class="project-links">
            <a href="#" class="btn btn-small">Live Demo</a>
            <a href="#" class="btn btn-small btn-outline">View Code</a>
        </div>
    </div>
</div>
```

### 4. Skills Section
Update your skills and proficiency levels:

```html
<div class="skill-item">
    <span>Your Skill</span>
    <div class="skill-bar">
        <div class="skill-progress" style="width: 90%"></div>
    </div>
</div>
```

### 5. Social Links
Update social media links in the contact section:

```html
<div class="social-links">
    <a href="https://github.com/yourusername" class="social-link">
        <i class="fab fa-github"></i>
    </a>
    <a href="https://linkedin.com/in/yourusername" class="social-link">
        <i class="fab fa-linkedin"></i>
    </a>
    <!-- Add more social links -->
</div>
```

## 🌐 Deployment Options

### 1. GitHub Pages
1. Push your code to a GitHub repository
2. Go to repository Settings → Pages
3. Select source branch (usually `main` or `gh-pages`)
4. Your site will be available at `https://yourusername.github.io/repository-name`

### 2. Netlify
1. Drag and drop your project folder to [Netlify](https://netlify.com)
2. Or connect your GitHub repository for automatic deployments

### 3. Vercel
1. Install Vercel CLI: `npm install -g vercel`
2. Run `vercel` in your project directory
3. Follow the deployment prompts

### 4. Traditional Web Hosting
Upload all files to your web hosting provider's public folder (usually `public_html` or `www`)

## 🔧 Advanced Features

### Contact Form Integration
To make the contact form functional, integrate with:
- **Formspree**: Simple form handling service
- **Netlify Forms**: Built-in form handling for Netlify sites
- **EmailJS**: Send emails directly from JavaScript
- **Custom Backend**: PHP, Node.js, or Python backend

Example with Formspree:
```html
<form action="https://formspree.io/f/your-form-id" method="POST" class="contact-form">
    <!-- form fields -->
</form>
```

### Analytics Integration
Add Google Analytics or other tracking:
```html
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_MEASUREMENT_ID');
</script>
```

### Performance Optimization
- **Image Optimization**: Use WebP format for images
- **Lazy Loading**: Already implemented for future images
- **Minification**: Minify CSS and JavaScript for production
- **CDN**: Use CDN for external libraries

## 📱 Browser Support

| Browser | Version |
|---------|---------|
| Chrome  | ✅ Latest |
| Firefox | ✅ Latest |
| Safari  | ✅ Latest |
| Edge    | ✅ Latest |
| IE      | ❌ Not supported |

## 🐛 Troubleshooting

### Common Issues

1. **Animations not working**
   - Check if JavaScript is enabled
   - Ensure CSS animations are supported

2. **Mobile navigation not working**
   - Verify JavaScript is loaded
   - Check for console errors

3. **Form not submitting**
   - Add proper form action and method
   - Implement backend form handling

4. **Fonts not loading**
   - Check internet connection
   - Verify Google Fonts URL

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Font Awesome** for the amazing icons
- **Google Fonts** for the beautiful typography
- **Inspiration** from modern portfolio designs
- **Color Palette** inspired by modern design trends

## 📞 Support

If you have any questions or need help customizing the portfolio:

1. **Check the Documentation**: Review this README thoroughly
2. **Search Issues**: Look for similar issues in the repository
3. **Create an Issue**: Open a new issue with detailed information
4. **Contact**: Reach out via the contact form on the portfolio

---

## 🚀 Featured Project: AI Attendance System

This portfolio prominently features an AI-powered attendance system with the following components:

### Project Files
- `1_datasetCreation.py` - Creates face recognition dataset
- `2_preprocessingEmbeddings.py` - Processes images for training
- `3_trainingFaceML.py` - Trains the machine learning model
- `5_recognizingPersonwithCSVDatabase.py` - Real-time recognition system

### Technologies Used
- **Python** - Core programming language
- **OpenCV** - Computer vision operations
- **OpenFace** - Face recognition neural network
- **Machine Learning** - Training and inference
- **CSV Database** - Attendance data storage

### Key Features
- ✅ Real-time face detection and recognition
- ✅ Automated dataset creation and preprocessing
- ✅ ML model training with high accuracy
- ✅ CSV-based attendance database

This project demonstrates practical application of AI and machine learning in solving real-world problems.

---

**Made with ❤️ for the AI & ML community**