# My utility codes
A variety of scripts implementing different aspects which I use from time to time in my coding. It's like my storage for some implementations I find intersting and generally useful.

![image](https://github.com/user-attachments/assets/3a0177af-16a9-4b7c-8fad-0957f9550724)

---
# 🧠  Training Checklist

This is a structured checklist to guide training an object detection model.

## 🔧 A) Training Configuration

1. **Batch Size**
2. **Number of Epochs**
3. **Learning Rate (`lr`)**
4. ⏱️ **Track Training Time**
5. 🎛️ **(For Generalization): Data Augmentation**
6. 📊 **Loss & Evaluation Logging + Visualization**

---

## ⚙️ B) Optimization & Evaluation

7. **Optimizer**
8. **Loss Function (Criterion)**
9. 🛑 **Early Stopping**
10. 🖨️ **Print Evaluation Metrics**
11. 🧪 **Split Strategy:** `train` / `val` / `test`
12. 💾 **Checkpointing (Saving Models):**
    - ✅ Best model (based on validation/test performance)
    - 📌 Save model every *n* epochs or last epoch

---

> Tip: Combine this checklist with tools like TensorBoard or Weights & Biases for better monitoring.
