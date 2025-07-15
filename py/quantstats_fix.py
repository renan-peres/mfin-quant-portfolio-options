#!/usr/bin/env python3
"""
QuantStats Compatibility Tool

This script resolves two common issues with QuantStats in CI environments:
1. Fixes pandas resampling issues by removing unsupported 'axis=0' parameter
2. Patches IPython to work in non-interactive environments by directly modifying the package files

Usage:
    python quantstats_fix.py
"""

import os
import re
import sys
import site
import shutil
import importlib
import importlib.util
from pathlib import Path
from typing import List, Optional, Tuple
import logging

def find_quantstats_paths() -> List[str]:
    """Find possible installation paths for QuantStats core plotting module."""
    quantstats_core_paths = []
    
    # Standard site-packages locations
    for site_dir in site.getsitepackages():
        potential_path = os.path.join(site_dir, 'quantstats', '_plotting', 'core.py')
        if os.path.exists(potential_path):
            quantstats_core_paths.append(potential_path)
    
    # User site-packages
    user_site = site.getusersitepackages()
    potential_path = os.path.join(user_site, 'quantstats', '_plotting', 'core.py')
    if os.path.exists(potential_path):
        quantstats_core_paths.append(potential_path)
    
    # Virtual environment if detected
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        venv_site_packages = os.path.join(sys.prefix, 'lib', f'python{py_version}', 'site-packages')
        potential_path = os.path.join(venv_site_packages, 'quantstats', '_plotting', 'core.py')
        if os.path.exists(potential_path):
            quantstats_core_paths.append(potential_path)
    
    # Look for the file in the current directory structure as well
    for root, _, files in os.walk(os.path.dirname(os.path.abspath(__file__))):
        if 'core.py' in files and 'quantstats' in root and '_plotting' in root:
            quantstats_core_paths.append(os.path.join(root, 'core.py'))
    
    return list(set(quantstats_core_paths))  # Remove duplicates


def fix_file_content(content: str) -> Tuple[Optional[str], List[str]]:
    """
    Fix resampling issues in file content by removing axis=0 parameter.
    
    Returns:
        Tuple containing:
        - Modified content (or None if no changes)
        - List of applied fixes for reporting
    """
    applied_fixes = []
    
    # Common pattern fixes
    patterns = [
        (r'returns = returns\.last\(\) if compound is True else returns\.sum\(axis=0\)', 
         'returns = returns.last() if compound is True else returns.sum()'),
        (r'benchmark = benchmark\.last\(\) if compound is True else benchmark\.sum\(axis=0\)', 
         'benchmark = benchmark.last() if compound is True else benchmark.sum()'),
        (r'\.sum\(axis=0\)', '.sum()')  # More generic pattern
    ]
    
    new_content = content
    for pattern, replacement in patterns:
        temp_content = re.sub(pattern, replacement, new_content)
        if temp_content != new_content:
            applied_fixes.append(f"Applied fix: {pattern} → {replacement}")
            new_content = temp_content
    
    # Check if we made any changes
    return (new_content if new_content != content else None, applied_fixes)


def find_package_path(package_name):
    """Find the installation path of a package"""
    try:
        spec = importlib.util.find_spec(package_name)
        if spec is not None and spec.origin:
            return os.path.dirname(spec.origin)
        
        # Try another approach if the above fails
        import pkg_resources
        return pkg_resources.resource_filename(package_name, '')
    except (ImportError, Exception):
        return None

def patch_quantstats_utils_file():
    """Directly patch the quantstats utils.py file"""
    utils_path = find_package_path("quantstats")
    if utils_path:
        # Get the actual file path for __init__.py
        utils_file = os.path.join(utils_path, "__init__.py")
            
        if os.path.exists(utils_file):
            print(f"Found QuantStats utils file at: {utils_file}")
            
            try:
                # Read the content
                with open(utils_file, 'r') as f:
                    lines = f.readlines()
                
                # Create backup
                backup_file = utils_file + '.bak'
                if not os.path.exists(backup_file):
                    print(f"Creating backup at: {backup_file}")
                    shutil.copy2(utils_file, backup_file)
                
                # Look for the problematic line and fix it
                fixed_lines = []
                i = 0
                fixed = False
                
                while i < len(lines):
                    line = lines[i]
                    
                    # Check for try: statement without proper indentation in next line
                    if "try:" in line.strip() and i < len(lines)-1:
                        # This is the try line, get its indentation
                        indentation = line[:len(line) - len(line.lstrip())]
                        try_line = line
                        
                        # Check next line
                        next_line = lines[i+1]
                        
                        # If the next line is the utils._in_notebook but doesn't have proper indentation
                        if "utils._in_notebook(" in next_line and len(next_line) - len(next_line.lstrip()) <= len(indentation):
                            # This is the problem - fix by adding all lines with proper indentation
                            fixed_lines.append(try_line)  # Add the try: line
                            
                            # Add properly indented notebook line
                            fixed_lines.append(f"{indentation}    utils._in_notebook(matplotlib_inline=True)\n")
                            
                            # Add except block
                            fixed_lines.append(f"{indentation}except AttributeError:\n")
                            fixed_lines.append(f"{indentation}    # Fall back to direct matplotlib configuration\n")
                            fixed_lines.append(f"{indentation}    import matplotlib\n")
                            fixed_lines.append(f"{indentation}    matplotlib.use('Agg')\n")
                            
                            # Skip the original problematic notebook line
                            i += 2
                            fixed = True
                            continue
                    
                    # Check for utils._in_notebook without a try/except
                    elif "utils._in_notebook(matplotlib_inline=True)" in line and "try:" not in line:
                        # Get the indentation of the current line
                        indentation = line[:len(line) - len(line.lstrip())]
                        
                        # Replace with properly structured try/except
                        fixed_lines.append(f"{indentation}try:\n")
                        fixed_lines.append(f"{indentation}    utils._in_notebook(matplotlib_inline=True)\n")
                        fixed_lines.append(f"{indentation}except AttributeError:\n")
                        fixed_lines.append(f"{indentation}    # Fall back to direct matplotlib configuration\n")
                        fixed_lines.append(f"{indentation}    import matplotlib\n")
                        fixed_lines.append(f"{indentation}    matplotlib.use('Agg')\n")
                        
                        fixed = True
                        i += 1
                        continue
                    
                    # No issues, keep the line
                    fixed_lines.append(line)
                    i += 1
                
                # Write the fixed content back to the file
                if fixed:
                    with open(utils_file, 'w') as f:
                        f.writelines(fixed_lines)
                    print("Successfully fixed indentation in QuantStats __init__.py file")
                    return True
                else:
                    print("No indentation issues found in QuantStats __init__.py file")
                    return True
                    
            except Exception as e:
                print(f"Error while patching file: {e}")
                return False
        else:
            print(f"QuantStats __init__.py file not found at expected location: {utils_file}")
            return False
    else:
        print("Could not find quantstats package path")
        return False

def check_for_resampling_issues(file_path: str) -> None:
    """Check file for potential resampling issues and print diagnostic info."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Look for the plot_timeseries function
    if 'def plot_timeseries' in content:
        print(f"✓ Found 'plot_timeseries' function in {file_path}")
        
        # Extract the function to examine
        match = re.search(r'def plot_timeseries.*?def ', content, re.DOTALL)
        if match:
            func_content = match.group(0)[:-4]  # Remove the trailing 'def '
            
            # Look for potential issues
            if 'sum(axis=0)' in func_content:
                print(f"⚠️ Found 'sum(axis=0)' in the function - likely needs fix")
                # Show the context
                for line in func_content.split('\n'):
                    if 'sum(axis=0)' in line:
                        print(f"  → {line.strip()}")
            else:
                print("✓ No 'sum(axis=0)' calls found - may already be fixed")
    else:
        print(f"⚠️ Could not find 'plot_timeseries' function in {file_path}")


def fix_quantstats_resampling_issue() -> bool:
    """
    Fix QuantStats resampling issue by modifying core.py file.
    
    Returns:
        bool: True if the fix was applied, False otherwise.
    """
    quantstats_core_paths = find_quantstats_paths()
    
    if not quantstats_core_paths:
        print("Could not find quantstats installation. Please provide the path manually.")
        try:
            manual_path = input("Enter the full path to quantstats/_plotting/core.py: ")
            if os.path.exists(manual_path):
                quantstats_core_paths.append(manual_path)
            else:
                print(f"Path {manual_path} does not exist.")
                return False
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled.")
            return False
    
    files_fixed = 0
    for file_path in quantstats_core_paths:
        try:
            print(f"Examining {file_path}...")
            
            # Add diagnostic check
            check_for_resampling_issues(file_path)
            
            # Read the file
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Fix the patterns
            new_content, applied_fixes = fix_file_content(content)
            
            # Check if any changes were made
            if new_content:
                # Create a backup of the original file
                backup_path = file_path + '.bak'
                with open(backup_path, 'w') as f:
                    f.write(content)
                print(f"Created backup at {backup_path}")
                
                # Write the fixed content
                with open(file_path, 'w') as f:
                    f.write(new_content)
                
                print(f"Successfully fixed {file_path}")
                for fix in applied_fixes:
                    print(f"  • {fix}")
                files_fixed += 1
            else:
                print(f"No applicable fixes found for {file_path}")
        
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    return files_fixed > 0

def fix_html_report():
    """Fix the HTML report functionality by adding a download button and removing attribution"""
    try:
        import quantstats as qs
        
        # Store the original HTML function
        original_html = qs.reports.html
        
        def fixed_html(*args, **kwargs):
            """Fixed version that adds a download button and removes QuantStats attribution from the saved HTML file"""
            output = kwargs.get('output')
            download_filename = kwargs.get('download_filename', 'tearsheet.html')
            
            if output:
                # First call the original function to generate the file
                original_html(*args, **kwargs)
                
                # Now modify the HTML file to add a download button and remove attribution
                if os.path.exists(output):
                    with open(output, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    
                    # Remove the QuantStats attribution line using regex
                    import re
                    pattern = r'<h4>.*?Generated by <a href="http://quantstats\.io" target="quantstats">QuantStats</a> \(v\. [0-9\.]+\)</h4>'
                    html_content = re.sub(pattern, '''<h4>Benchmark Selection: <a href="https://github.com/renan-peres/mfin-portfolio-management/blob/main/03_benchmark_selection.ipynb" target="_blank">03_benchmark_selection.ipynb</a></h4>
                            <h4>Benchmark Comparison: <a href="https://github.com/renan-peres/mfin-portfolio-management/blob/main/reports/01_benchmark_comparison_quantstats.ipynb" target="_blank">01_benchmark_comparison_quantstats.ipynb</a></h4>
                            <hr>''', html_content)
                    
                    # Create a download button that saves the current page - now in top right
                    download_js = """
                    <div style="text-align:center;margin:20px 0;position:fixed;top:20px;right:20px;z-index:1000;">
                      <button onclick="downloadHTML()" style="padding:10px 15px;background:#4CAF50;color:white;border:none;border-radius:4px;cursor:pointer;font-size:16px;">
                        Download Report
                      </button>
                    </div>
                    <script>
                    function downloadHTML() {
                        var a = document.createElement('a');
                        a.href = "data:text/html;charset=utf-8," + encodeURIComponent(document.documentElement.outerHTML);
                        a.download = "%s";
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                    }
                    </script>
                    """ % download_filename
                    
                    # Try multiple insertion points for better compatibility
                    modified_html = html_content
                    
                    # Check if there's an existing body tag with attributes
                    if '</head>' in html_content:
                        modified_html = html_content.replace('</head>', '</head>' + download_js, 1)
                    # Before end of body
                    elif '</body>' in html_content:
                        modified_html = html_content.replace('</body>', download_js + '</body>', 1)
                    # Before end of HTML
                    elif '</html>' in html_content:
                        modified_html = html_content.replace('</html>', download_js + '</html>', 1)
                    # Last resort - just append at the end
                    else:
                        modified_html = html_content + download_js
                    
                    # Write the modified HTML back to the file
                    with open(output, 'w', encoding='utf-8') as f:
                        f.write(modified_html)
                    
                    print(f"Added download button and removed QuantStats attribution from {output}")
            else:
                original_html(*args, **kwargs)
        
        # Replace the original function with our fixed version
        qs.reports.html = fixed_html
        print("Successfully patched QuantStats HTML report functionality")
        return True
    except Exception as e:
        print(f"Failed to patch HTML report: {str(e)}")
        return False

# Apply fixes in the correct order
print("============================================================")
print("               QuantStats Compatibility Tool                ")
print("============================================================")

print("\nPart 1: Directly patching QuantStats package files")
print("------------------------------------------------------------")
utils_patched = patch_quantstats_utils_file()
if utils_patched:
    print("✓ QuantStats utils file patched successfully")
else:
    print("⚠️ Failed to patch QuantStats utils file")

print("\nPart 2: Fixing resampling issues")
print("------------------------------------------------------------")
# First check the quantstats paths
quantstats_core_paths = find_quantstats_paths()
if quantstats_core_paths:
    print(f"Found {len(quantstats_core_paths)} potential QuantStats installation(s)")
    for path in quantstats_core_paths:
        print(f"Checking {path}")
        check_for_resampling_issues(path)
else:
    print("No QuantStats installations found to check")

# Now apply the resampling fix
resampling_fixed = fix_quantstats_resampling_issue()
if resampling_fixed:
    print("✓ Resampling issues fixed successfully")
else:
    print("⚠️ Resampling fix not applied or not needed")

# Now it's safe to import quantstats
print("\nPart 3: Importing and patching QuantStats")
print("------------------------------------------------------------")
try:
    # Force reload in case it was already imported
    if "quantstats" in sys.modules:
        print("Reloading quantstats module...")
        importlib.reload(sys.modules["quantstats"])
        if "quantstats.utils" in sys.modules:
            importlib.reload(sys.modules["quantstats.utils"])
    
    # Now import
    import quantstats as qs
    print("✓ Successfully imported quantstats")

    # Apply HTML report fix
    html_fixed = fix_html_report()
    if html_fixed:
        print("✓ HTML report functionality enhanced with download button")
    else:
        print("⚠️ HTML report enhancement not applied")
        
except Exception as e:
    print(f"❌ Error importing quantstats: {str(e)}")
    
if __name__ == "__main__":
    print("\nAll fixes have been applied. QuantStats should now work in both interactive and non-interactive environments.")
    print("You can now use QuantStats in your analysis scripts.")