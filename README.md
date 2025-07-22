# My utility codes
A variety of scripts implementing different aspects which I use from time to time in my coding. It's like my storage for some implementations I find intersting and generally useful.

![image](https://github.com/user-attachments/assets/3a0177af-16a9-4b7c-8fad-0957f9550724)

---
# 🧠  Training Checklist

This is a structured checklist to guide training an object detection model.

## 🔧 A) Training Configuration

0. BE CAREFUL WHERE YOU STORE THE DATA:  SSD ALWAYS!!! (with HDD its 10 x slower)
1. **Batch Size**
2. **Number of Epochs**
3. **Learning Rate (`lr`)**
4. ⏱️ **Track Training Time** (tqdm does it also)
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
13. A good idea is to use all the hyperparameters as parser.arguments, and then save the command also in the loggings! 
    So that way you will know the results, but also how you got them.:
        
    with open(log_path_of_txt_file, 'a') as f:
        f.write(f'command used: \n      python3 {" ".join(sys.argv)}\n')

14. Implement the loading from saved checkpoint (start_epoch, epochs) from the beginning because your training might crash.
    Save model, optimizer, epoch, best_metric.

15. Use torch.compile(model) afters creating your model. It optimizes the training. But when you save the model, save the un-optimized.
    Meaning: optimized_model = torch.compile(model) , but torch.save(model.state_dict, "ckpt.pt") !!!
---

> Tip: Combine this checklist with tools like TensorBoard or Weights & Biases for better monitoring.
