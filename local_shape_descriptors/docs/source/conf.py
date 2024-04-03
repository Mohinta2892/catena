# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Catena"
copyright = "2023, Samia Mohinta"
author = "Samia Mohinta"
release = "4-Sep-2023"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx_copybutton"]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'sphinx_rtd_theme'
# html_theme = "furo"
# html_theme_options = {
#    "light_css_variables": {
#        "color-brand-primary": "darkgray",
#        "color-brand-content": "#CC3333",
#        "color-admonition-background": "orange",
#        "font-size": 12
#    },
# }
html_theme = "pydata_sphinx_theme"

html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
