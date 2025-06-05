import subprocess
import sys
import pkg_resources

def install_requirements():
    try:
        # Read requirements.txt
        with open('requirements.txt', 'r') as f:
            required = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        # Check installed packages
        installed = {pkg.key for pkg in pkg_resources.working_set}
        
        # Install missing packages
        for package in required:
            pkg_name = package.split('==')[0].lower()
            if pkg_name not in installed:
                print(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            else:
                print(f"{pkg_name} is already installed.")
                
        print("All dependencies installed successfully.")
    
    except FileNotFoundError:
        print("Error: requirements.txt not found in the current directory.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    install_requirements()