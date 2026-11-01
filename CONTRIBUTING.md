# Contributing to Typhon IDS

Thank you for your interest in contributing to the **Typhon Intrusion Detection System**! We welcome contributions ranging from bug fixes and documentation improvements to new neural network modules, alternative tensor contraction backends, and benchmark integrations.

---

## 1. Development Workflow

1. **Fork the Repository**:
   Click the "Fork" button at the top right of the GitHub repository page.

2. **Clone your Fork**:
   ```bash
   git clone https://github.com/<your-username>/typhon.git
   cd typhon
   ```

3. **Create a Dedicated Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Set Up a Clean Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate

   pip install -r requirements.txt
   ```

5. **Make and Test Changes**:
   Verify that your modifications do not break existing detection pipelines:
   ```bash
   python main.py --mode detect --file data/sample_flows.csv
   ```

6. **Commit & Push**:
   Write clear, semantic commit messages:
   ```bash
   git commit -m "feat(ct_dgnn): add multi-head message passing support"
   git push origin feature/your-feature-name
   ```

7. **Submit a Pull Request (PR)**:
   Open a pull request describing the rationale behind your changes and relevant benchmark performance.

---

## 2. Coding Standards

- **PEP 8**: Follow PEP 8 guidelines for Python code style.
- **Type Annotations**: Provide type annotations for function signatures where possible (`typing.Dict`, `typing.List`, `torch.Tensor`, etc.).
- **Reproducibility**: Set random seeds (`torch.manual_seed(42)`, `np.random.seed(42)`) in training and evaluation scripts.
- **Documentation**: Update docstrings and corresponding markdown documentation in `docs/` when introducing new config flags or model layers.

---

## 3. Reporting Issues

If you find a bug or performance degradation:
- Open an Issue with a descriptive title.
- Provide your OS, Python version, PyTorch version, and hardware configuration (CPU / GPU / RAM).
- Include the terminal command executed and the full traceback.
