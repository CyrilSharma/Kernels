from invoke import task
import os
import subprocess
import sys

# Ensure we're in the right directory
BUILD_DIR = "build"

@task
def build(ctx, target="all"):
    """Build a target using the generated Makefile"""
    if not os.path.exists(BUILD_DIR):
        print("Build directory doesn't exist. Run 'inv configure' first.")
        return
    
    os.chdir(BUILD_DIR)
    try:
        ctx.run(f"make {target}")
    finally:
        os.chdir("..")

@task
def test(ctx, target=None):
    """Run tests using the generated Makefile"""
    if not os.path.exists(BUILD_DIR):
        print("Build directory doesn't exist. Run 'inv configure' first.")
        return
    
    os.chdir(BUILD_DIR)
    try:
        if target:
            # Build and run specific test
            ctx.run(f"make {target}")
            ctx.run(f"./{target}")
        else:
            # Run all tests
            ctx.run("make test")
    finally:
        os.chdir("..")

@task
def configure(ctx):
    """Configure the project with CMake"""
    if not os.path.exists(BUILD_DIR):
        os.makedirs(BUILD_DIR)
    
    os.chdir(BUILD_DIR)
    try:
        ctx.run("cmake ..")
        print("Project configured successfully!")
    finally:
        os.chdir("..")

@task
def clean(ctx):
    """Clean build artifacts"""
    if os.path.exists(BUILD_DIR):
        os.chdir(BUILD_DIR)
        try:
            ctx.run("make clean")
        finally:
            os.chdir("..")
    else:
        print("Build directory doesn't exist.")

@task
def rebuild(ctx):
    """Clean and rebuild everything"""
    clean(ctx)
    configure(ctx)
    build(ctx)

@task
def run(ctx, target):
    """Build and run a specific target"""
    build(ctx, target)
    if os.path.exists(BUILD_DIR):
        os.chdir(BUILD_DIR)
        try:
            ctx.run(f"./{target}")
        finally:
            os.chdir("..")

@task
def help(ctx):
    """Show available make targets"""
    if not os.path.exists(BUILD_DIR):
        print("Build directory doesn't exist. Run 'inv configure' first.")
        return
    
    os.chdir(BUILD_DIR)
    try:
        ctx.run("make help")
    finally:
        os.chdir("..")

@task
def path(ctx, target):
    """Output the exact path of a test executable for profiling tools like ncu"""
    if not os.path.exists(BUILD_DIR):
        print("Build directory doesn't exist. Run 'inv configure' first.")
        return
    
    test_path = os.path.join(BUILD_DIR, target)
    if os.path.exists(test_path):
        print(test_path)
    else:
        print(f"Test executable '{target}' not found in {BUILD_DIR}")
        print("Available tests:")
        os.chdir(BUILD_DIR)
        try:
            for test_source in ["test-avg-pool-1d", "test-avg-pool-1d-simple"]:
                if os.path.exists(test_source):
                    print(f"  {test_source}")
        finally:
            os.chdir("..")
