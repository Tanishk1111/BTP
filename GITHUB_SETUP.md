# GitHub Setup Guide for SpatX

This guide will help you push the SpatX project to GitHub.

## 📋 Pre-Push Checklist

✅ `.gitignore` configured to exclude:

- `archive/` directory (all old/test code)
- `test_data/` directory
- `user_models/`, `uploads/`, `training_data/`, `results/` (user data)
- Database files (`*.db`, `*.sqlite`)
- Virtual environments (`.venv311/`)
- Large model files (`*.pth`, `*.pt`)
- Python cache (`__pycache__/`, `*.pyc`)
- Log files (`logs/*.log`)
- Old markdown docs and test scripts

✅ Essential files included:

- Core application code (`app_enhanced.py`, `app_training.py`, etc.)
- Frontend (`frontend_working/`)
- Model architecture code (`spatx_core/`)
- Configuration files (`.gitignore`, `requirements.txt`)
- Documentation (`README.md`)
- Startup scripts (`start_local.sh`, `start_local.ps1`, etc.)
- Directory placeholders (`.gitkeep` files)

## 🚀 Step-by-Step GitHub Push

### 1. Initialize Git (if not already done)

```bash
git init
```

### 2. Check what will be committed

```bash
git status
```

**Expected to be ignored** (should NOT show up):

- `archive/`
- `test_data/`
- `user_models/` (except `.gitkeep`)
- `uploads/` (except `.gitkeep`)
- `training_data/` (except `.gitkeep`)
- `*.db` files
- `.venv311/`
- `logs/*.log`
- `__pycache__/`

**Expected to be committed** (should show up):

- `app_enhanced.py`, `app_training.py`, `database.py`, `models.py`, `gene_metadata.py`
- `frontend_working/index.html`
- `spatx_core/` (code only, not large model files)
- `saved_models/` (model architecture scripts, not `.pth` files)
- `requirements.txt`
- `README.md`
- `start_local.sh`, `start_local.ps1`, `stop_local.sh`, `stop_local.ps1`
- `.gitignore`
- `.gitkeep` files

### 3. Add files to staging

```bash
# Add all files (respecting .gitignore)
git add .

# Or add selectively
git add app_enhanced.py app_training.py database.py models.py gene_metadata.py
git add frontend_working/
git add spatx_core/
git add requirements.txt README.md .gitignore
git add start_local.sh start_local.ps1 stop_local.sh stop_local.ps1
git add uploads/.gitkeep user_models/.gitkeep training_data/.gitkeep results/.gitkeep logs/.gitkeep
```

### 4. Verify what's staged

```bash
git status
```

Make sure:

- ❌ No `archive/` files
- ❌ No database files (`.db`)
- ❌ No large model files (`.pth`)
- ❌ No test data or user uploads

### 5. Create initial commit

```bash
git commit -m "Initial commit: SpatX Spatial Transcriptomics Platform

- Complete FastAPI backend with training and prediction
- Responsive frontend with Tailwind CSS
- CIT-based model for 50+ breast cancer genes
- User authentication with JWT
- Training pipeline with background jobs
- Advanced visualizations (heatmaps, contours, overlays)
"
```

### 6. Create GitHub repository

Go to https://github.com/new and create a new repository:

- Name: `spatx` (or your preferred name)
- Description: "AI-powered spatial transcriptomics platform using vision transformers"
- **Do NOT** initialize with README, .gitignore, or license (we already have them)
- Visibility: Public or Private (your choice)

### 7. Add remote and push

```bash
# Add GitHub remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/spatx.git

# Verify remote
git remote -v

# Push to GitHub
git branch -M main
git push -u origin main
```

### 8. Verify on GitHub

Visit your repository and check:

- ✅ Core files are present
- ✅ README displays correctly
- ✅ `archive/` is NOT present
- ✅ No database files
- ✅ Empty directories have `.gitkeep` files

## 🔄 Future Updates

After making changes:

```bash
# Check what changed
git status

# Add changed files
git add <files>

# Commit with descriptive message
git commit -m "Description of changes"

# Push to GitHub
git push
```

## 🌿 Working with Branches

For major features, create a branch:

```bash
# Create and switch to new branch
git checkout -b feature/new-feature

# Make changes, commit them
git add .
git commit -m "Add new feature"

# Push branch to GitHub
git push -u origin feature/new-feature

# On GitHub, create a Pull Request to merge into main
```

## 🛠️ Troubleshooting

### Files too large error

If you accidentally try to commit large files:

```bash
# Remove from staging
git reset HEAD <large-file>

# Add to .gitignore
echo "<large-file>" >> .gitignore

# Commit without the large file
git add .gitignore
git commit -m "Update gitignore"
```

### Already committed large files

If large files were already committed:

```bash
# Remove from git history (careful!)
git rm --cached <large-file>
git commit -m "Remove large file from tracking"
```

### Wrong files committed

```bash
# Undo last commit (keep changes)
git reset --soft HEAD~1

# Fix .gitignore and re-commit
```

## 📝 Recommended Repository Settings

After pushing to GitHub, configure:

1. **Branches** → Set `main` as default branch
2. **Settings** → Add description and topics:
   - Topics: `spatial-transcriptomics`, `deep-learning`, `vision-transformer`, `fastapi`, `bioinformatics`
3. **About** → Add website (if deployed)
4. **README** → Ensure badges are working (optional: add CI/CD badges later)

## 🔐 Private vs Public

**Public Repository:**

- ✅ Good for portfolio/resume
- ✅ Open source collaboration
- ⚠️ Make sure no secrets (API keys, passwords) are committed

**Private Repository:**

- ✅ Keep research private until publication
- ✅ Control access
- ⚠️ Still avoid committing secrets

## ⚠️ Security Checklist

Before making repository public:

- ✅ No hardcoded passwords or API keys
- ✅ `SECRET_KEY` in code is placeholder (users should change it)
- ✅ No real user data in test files
- ✅ No `.env` files with actual credentials
- ✅ Database files are gitignored

---

**Ready to push!** 🚀

Run the commands above to get your code on GitHub.
