#!/bin/bash
# Quick Git Commands for SpatX GitHub Push

echo "🔍 Step 1: Check current status"
echo "────────────────────────────────"
git status
echo ""

read -p "❓ Does this look correct? (archive/ should NOT be listed) [y/N]: " confirm
if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo "❌ Please review .gitignore and try again"
    exit 1
fi

echo ""
echo "📦 Step 2: Stage all files"
echo "────────────────────────────────"
git add .
git status
echo ""

read -p "❓ Ready to commit? [y/N]: " confirm
if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo "❌ Aborted"
    exit 1
fi

echo ""
echo "💾 Step 3: Create commit"
echo "────────────────────────────────"
git commit -m "Initial commit: SpatX Spatial Transcriptomics Platform

- Complete FastAPI backend with training and prediction
- Responsive frontend with Tailwind CSS  
- CIT-based model for 50+ breast cancer genes
- User authentication with JWT
- Training pipeline with background jobs
- Advanced visualizations (heatmaps, contours, overlays)
"

echo ""
echo "🔗 Step 4: Add remote (if not already added)"
echo "────────────────────────────────"
echo "Enter your GitHub repository URL:"
echo "Format: https://github.com/YOUR_USERNAME/spatx.git"
read -p "URL: " repo_url

if [ ! -z "$repo_url" ]; then
    git remote add origin "$repo_url" 2>/dev/null || echo "Remote already exists"
    git remote -v
fi

echo ""
echo "🚀 Step 5: Push to GitHub"
echo "────────────────────────────────"
read -p "❓ Push to GitHub now? [y/N]: " confirm
if [[ $confirm =~ ^[Yy]$ ]]; then
    git branch -M main
    git push -u origin main
    echo ""
    echo "✅ Successfully pushed to GitHub!"
    echo "🌐 Visit your repository to verify"
else
    echo "⏸️  Push skipped. Run manually: git push -u origin main"
fi

