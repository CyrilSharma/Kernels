from invoke import task, Context
import os
from functools import wraps

# Ensure we're in the right directory
BUILD_DIR = "build"

def requires_build_dir(create_if_missing=False):
    """Decorator to ensure build directory exists and run inside it"""
    def decorator(func):
        @wraps(func)
        def wrapper(ctx, *args, **kwargs):
            if not os.path.exists(BUILD_DIR):
                if create_if_missing:
                    os.makedirs(BUILD_DIR)
                    print(f"Created build directory: {BUILD_DIR}")
                else:
                    print("Build directory doesn't exist. Run 'inv configure' first.")
                    return
            with ctx.cd(BUILD_DIR):
                return func(ctx, *args, **kwargs)
        return wrapper
    return decorator

@task
@requires_build_dir()
def build(ctx, target="all"):
    """Build a target using the generated Makefile"""
    ctx.run(f"make {target}")

@task
@requires_build_dir()
def test(ctx, target=None):
    """Run tests using the generated Makefile"""
    if target:
        target = f"test-{target}"
        ctx.run(f"make {target}")
        ctx.run(f"./{target}")
    else:
        ctx.run("make test")

@task
@requires_build_dir(create_if_missing=True)
def configure(ctx):
    """Configure the project with CMake"""
    ctx.run("cmake ..")
    print("Project configured successfully!")

@task
@requires_build_dir()
def clean(ctx):
    """Clean build artifacts"""
    ctx.run("make clean")

@task
def rebuild(ctx):
    """Clean and rebuild everything"""
    clean(ctx)
    configure(ctx)
    build(ctx)

@task
@requires_build_dir()
def run(ctx, target):
    """Build and run a specific target"""
    build(ctx, target)
    ctx.run(f"./{target}")

@task
@requires_build_dir()
def help(ctx):
    """Show available make targets"""
    ctx.run("make help")

@task
@requires_build_dir()
def profile(ctx, target=None):
    """Build and run a profiling executable"""
    if target:
        target = f"profile-{target}"
        ctx.run(f"make {target}")
        ctx.run(f"./{target}")
    else:
        ctx.run("make profile")
