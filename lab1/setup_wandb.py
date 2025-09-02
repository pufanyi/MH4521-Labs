"""
Weights & Biases Setup Script for MH4521 Labs
This script helps you set up wandb for experiment tracking.
"""

import wandb
import os

def setup_wandb():
    """Set up Weights & Biases for experiment tracking."""
    print("Setting up Weights & Biases for MH4521 Labs...")
    print("="*50)
    
    # Check if wandb is already logged in
    try:
        api = wandb.Api()
        user = api.viewer
        print(f"✓ Already logged in as: {user.username}")
        print(f"✓ Entity: {user.entity}")
        return True
    except Exception:
        print("Not logged in to Weights & Biases.")
    
    print("\nTo use wandb experiment tracking, you need to:")
    print("1. Create a free account at https://wandb.ai")
    print("2. Get your API key from https://wandb.ai/authorize")
    print("3. Run 'wandb login' in your terminal")
    print("   OR")
    print("4. Set the WANDB_API_KEY environment variable")
    
    # Try to login interactively
    try:
        print("\nAttempting to login...")
        wandb.login()
        print("✓ Successfully logged in to Weights & Biases!")
        return True
    except Exception as e:
        print(f"✗ Login failed: {e}")
        print("\nAlternative setup options:")
        print("1. Run 'wandb login' in your terminal manually")
        print("2. Set WANDB_MODE=offline to track experiments locally")
        print("3. Set WANDB_MODE=disabled to disable tracking")
        return False

def test_wandb():
    """Test wandb setup with a simple experiment."""
    print("\nTesting wandb setup...")
    
    try:
        # Initialize a test run
        with wandb.init(project="mh4521-test", name="setup-test", mode="online") as run:
            # Log some test data
            for i in range(10):
                wandb.log({"test_metric": i * 0.1, "step": i})
            
            wandb.log({"test_completed": True})
            print("✓ Test experiment logged successfully!")
            print(f"✓ View your test run at: {run.url}")
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
        print("You can still run experiments in offline mode.")
        return False
    
    return True

def configure_offline_mode():
    """Configure wandb for offline mode."""
    print("\nConfiguring offline mode...")
    os.environ["WANDB_MODE"] = "offline"
    print("✓ Wandb configured for offline mode.")
    print("Your experiments will be saved locally and can be synced later.")
    print("To sync later, run: wandb sync <run_directory>")

if __name__ == "__main__":
    print("MH4521 Labs - Weights & Biases Setup")
    print("="*40)
    
    # Try to set up wandb
    if setup_wandb():
        if test_wandb():
            print("\n🎉 Wandb is ready to use!")
            print("\nNext steps:")
            print("1. Run 'python lab1/run.py' for a single experiment")
            print("2. Run 'python lab1/compare_agents.py' for multi-agent comparison")
        else:
            print("\n⚠️  Setup completed but test failed.")
            print("You can still run experiments - they might be saved offline.")
    else:
        print("\n💡 Consider running in offline mode for now.")
        response = input("Would you like to configure offline mode? (y/n): ")
        if response.lower().startswith('y'):
            configure_offline_mode()
    
    print("\nFor more information, see: https://docs.wandb.ai/quickstart")
