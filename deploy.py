import os
import shutil

def prepare_for_deployment():
    """
    Prepare files for GitHub Pages deployment
    """
    print("Preparing for GitHub Pages deployment...")
    
    # Files needed for GitHub Pages
    github_pages_files = [
        'index.html',
        'README.md',
        'LICENSE'
    ]
    
    # Create a docs folder for GitHub Pages
    if os.path.exists('docs'):
        shutil.rmtree('docs')
    os.makedirs('docs')
    
    # Copy necessary files
    for file in github_pages_files:
        if os.path.exists(file):
            shutil.copy(file, 'docs/')
            print(f"Copied {file} to docs/")
    
    # Copy some sample images
    sample_images = ['download.jpg', 'mole.jpg', 'ISIC_0000019_downsampled.jpg']
    for img in sample_images:
        if os.path.exists(img):
            shutil.copy(img, 'docs/')
            print(f"Copied {img} to docs/")
    
    print("Deployment preparation complete!")
    print("Push the 'docs' folder to GitHub and enable GitHub Pages from docs/ folder")

if __name__ == "__main__":
    prepare_for_deployment()