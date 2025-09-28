# Development of a Web-Based Platform for AI-Driven Spatial Transcriptomics Analysis from Histopathology Images

**A PROJECT REPORT**  
Submitted as part of BBD451 B.Tech Major Project (BB1)

**Submitted by:** Tanishk 2022BB11400

**Guided by:** Prof. Ishaan Gupta

---

**DEPARTMENT OF BIOCHEMICAL ENGINEERING AND BIOTECHNOLOGY**  
**INDIAN INSTITUTE OF TECHNOLOGY DELHI**  
**November, 2025**

---

## DECLARATION

I certify that:

a) the work contained in this report is original and has been done by me under the guidance of my supervisor(s).

b) I have followed the guidelines provided by the Department in preparing the report.

c) I have conformed to the norms and guidelines given in the Honor Code of Conduct of the Institute.

d) whenever I have used materials (data, theoretical analysis, figures, and text) from other sources, I have given due credit to them by citing them in the text of the report and giving their details in the references.

**Signed by:** Tanishk 2022BB11400

---

## CERTIFICATE

It is certified that the work contained in this report titled "Development of a Web-Based Platform for AI-Driven Spatial Transcriptomics Analysis from Histopathology Images" is the original work done by Tanishk (2022BB11400) and has been carried out under my supervision.

**Signed by:** Prof. Ishaan Gupta  
**Date:** **_/_**/2025

---

## ABSTRACT

Spatial transcriptomics (ST) provides critical insights into cellular function within the context of tissue architecture but remains prohibitively expensive and technically demanding for widespread use. Recent advancements in deep learning, particularly models like HisToGene and ST-Net, have demonstrated the potential to predict spatial gene expression directly from standard, low-cost histology images. However, these powerful models lack an accessible interface for researchers and clinicians.

This project addresses this gap by developing a comprehensive, user-friendly web platform. The primary objective is to implement a full-stack website that allows users to upload histopathology images and receive predicted spatial transcriptomics data, effectively lowering the barrier to entry for this technology. The platform leverages the spatx_core library, a custom-built computational engine that interfaces with the underlying prediction models.

By providing an intuitive interface, this work aims to democratize access to ST analysis, potentially reducing costs by an order of magnitude, and creating a framework for gathering new data to further refine and improve the predictive models.

---

## TABLE OF CONTENTS

- Declaration ............................................................................................. ii
- Certificate ............................................................................................. iii
- Abstract ................................................................................................ iv
- List of Figures ......................................................................................... v
- List of Abbreviations ............................................................................... vi
- Chapter 1: Introduction ............................................................................ 1
  - 1.1 Background and Motivation .............................................................. 1
  - 1.2 Project Objectives ........................................................................... 2
- Chapter 2: Literature Survey ..................................................................... 3
  - 2.1 Spatial Transcriptomics ................................................................... 3
  - 2.2 AI in Computational Pathology ......................................................... 3
  - 2.3 Foundational Models: HisToGene and ST-Net ................................... 4
- Chapter 3: Methodology and Work Done .................................................... 5
  - 3.1 System Architecture ........................................................................ 5
  - 3.2 The spatx_core Library .................................................................. 5
  - 3.3 Web Application Development .......................................................... 6
  - 3.4 Development Environment ............................................................... 6
- Chapter 4: Initial Results and Progress ...................................................... 7
  - 4.1 Front-End User Interface ................................................................ 7
  - 4.2 Back-End API Implementation .......................................................... 7
  - 4.3 Technical Implementation Details ..................................................... 8
- Chapter 5: Conclusion and Future Work ..................................................... 9
  - 5.1 Summary of Progress ..................................................................... 9
  - 5.2 Future Work Plan .......................................................................... 9
- References ............................................................................................ 10

---

## List of Figures

**[Space reserved for figure insertions after completion]**

- Figure 3.1: High-level architecture of the web platform ................................. 5
- Figure 4.1: Screenshot of the main user interface dashboard .......................... 7
- Figure 4.2: Image upload interface and file validation process ........................ 7
- Figure 4.3: Results visualization page showing predicted gene expression ........ 8
- Figure 4.4: Backend API architecture and data flow diagram .......................... 8

---

## List of Abbreviations

- **AI:** Artificial Intelligence
- **API:** Application Programming Interface
- **H&E:** Hematoxylin and Eosin
- **HTML:** HyperText Markup Language
- **ST:** Spatial Transcriptomics
- **UI:** User Interface
- **REST:** Representational State Transfer
- **JSON:** JavaScript Object Notation
- **ML:** Machine Learning
- **CNN:** Convolutional Neural Network
- **GPU:** Graphics Processing Unit

---

# Chapter 1: Introduction

## 1.1 Background and Motivation

Spatial transcriptomics (ST) is a revolutionary technology that allows scientists to measure gene activity within a tissue sample while preserving spatial information. This provides unprecedented insight into cell-cell interactions, tissue heterogeneity, and disease pathology. However, the high cost and complex protocols associated with current ST methods limit their application in both research and clinical settings.

Concurrently, deep learning has made significant strides in the field of computational pathology. Standard Hematoxylin and Eosin (H&E) stained histology slides are inexpensive and routinely produced in clinics worldwide. Recent research, such as the HisToGene and ST-Net models, has shown that it is possible to train complex neural networks to predict spatial gene expression patterns directly from these H&E images. This AI-driven approach has the potential to make spatial genomics accessible, scalable, and over ten times more cost-effective.

While these models are powerful, they remain confined to computational experts who can run complex code. There is a critical need for a tool that bridges the gap between these advanced models and the biomedical researchers or clinicians who stand to benefit most from them. This project is motivated by the need to democratize this cutting-edge technology by creating an accessible, web-based platform. The platform will not only serve as a tool for the broader scientific community but also as a mechanism to gather valuable data, which can be used to further enhance the underlying AI models.

## 1.2 Project Objectives

The primary goal of this B.Tech project is to design, develop, and implement a full-stack web application for spatial transcriptomics prediction. The specific objectives for the project are:

• To understand the theoretical basis of the HisToGene and ST-Net models for predicting gene expression from histology.

• To develop a robust back-end server that integrates the spatx_core library to process user-submitted images and run the prediction model.

• To design and build an intuitive front-end user interface (UI) for image uploading and visualization of results.

• To establish a data-handling pipeline that securely manages user data and prepares it for potential future model retraining and improvement.

• To create a functional prototype of the web platform that demonstrates the core functionality of the prediction pipeline.

• To implement proper error handling and user feedback mechanisms for a seamless user experience.

---

# Chapter 2: Literature Survey

This project is built upon the intersection of spatial genomics and artificial intelligence. The foundational work in these fields provides the basis for the model and the motivation for the platform's development.

## 2.1 Spatial Transcriptomics

Spatial transcriptomics technologies have evolved rapidly, enabling the mapping of the transcriptome with spatial resolution. Techniques like 10x Genomics Visium are powerful but require specialized equipment and reagents, contributing to their high cost. The ability to extract similar information from standard histology slides represents a paradigm shift, as described by He et al. (2024) in their work on a visual-omics foundation model. Their work highlights the potential to bridge the gap between widely available histopathology data and complex spatial transcriptomics data.

## 2.2 AI in Computational Pathology

The integration of artificial intelligence in pathology has shown remarkable progress in recent years. Deep learning models have demonstrated exceptional performance in image classification, object detection, and pattern recognition tasks on histopathological images. The availability of large datasets and advances in computational power have enabled the development of sophisticated models capable of extracting complex features from tissue images.

## 2.3 Foundational Models: HisToGene and ST-Net

The predictive power of our platform relies on models inspired by recent breakthroughs. The ST-Net project introduced a graph neural network-based method for predicting spatial expression, demonstrating the feasibility of this approach. It established a framework for integrating spatial relationships within histology images to infer gene activity.

Building on this, the HisToGene model further refined this concept, likely employing advanced architectures like Vision Transformers or attention mechanisms to improve prediction accuracy and generalize across different tissue types. These models serve as the "brains" of our platform, and understanding their architecture is crucial for successful integration via the spatx_core library.

---

# Chapter 3: Methodology and Work Done

The development of the web platform involves a full-stack approach, integrating a machine learning core with a user-facing application.

## 3.1 System Architecture

The platform is designed with a modern client-server architecture. The user interacts with a React-based front-end, which communicates with a FastAPI back-end server via REST API calls. The back-end is responsible for processing requests, running the predictive model through the spatx_core library, and returning the results to the user.

**[Space reserved for Figure 3.1: System Architecture Diagram]**

The architecture consists of the following components:

- **Frontend:** React/TypeScript application with modern UI components
- **Backend:** FastAPI server with Python-based prediction pipeline
- **Model Layer:** spatx_core library interfacing with pre-trained models
- **Data Storage:** File management system for uploaded images and results

## 3.2 The spatx_core Library

The core of the application's predictive capability is handled by the spatx_core library, developed by Purushottam Singh. This Python library serves as a standardized interface to the underlying PyTorch-based machine learning models (HisToGene/ST-Net). My work involved integrating this library into the back-end, creating functions to load the pre-trained model, preprocess the input image data into the format required by the model, and execute the prediction pipeline.

Key integration features include:

- Model loading and initialization
- Image preprocessing pipelines
- Batch processing capabilities
- Error handling and validation

## 3.3 Web Application Development

The website is being developed using modern web technologies:

**Back-End:** The back-end is built using the FastAPI framework, chosen for its high performance and automatic API documentation generation. I have implemented API endpoints for:

- Image upload and validation
- Prediction processing
- Results retrieval
- File management

**Front-End:** The front-end is developed using React with TypeScript, providing a type-safe and maintainable codebase. Key features include:

- Responsive design for multiple device types
- Drag-and-drop file upload interface
- Real-time progress indicators
- Interactive results visualization

## 3.4 Development Environment

The development setup includes:

- **Local Development:** VS Code with Python and Node.js environments
- **Version Control:** Git repository with structured branching
- **Testing Environment:** Local servers for development and testing
- **Deployment Target:** Lab server environment (Ubuntu 18.04.6 LTS)

---

# Chapter 4: Initial Results and Progress

As of the mid-semester evaluation, significant progress has been made in setting up the foundational components of the web application.

## 4.1 Front-End User Interface

A functional and intuitive user interface has been developed using React and modern CSS frameworks. The interface includes:

**Main Dashboard:**

- Clean, professional design with clear navigation
- Responsive layout that works on desktop and mobile devices
- Integration with modern UI components for enhanced user experience

**[Space reserved for Figure 4.1: Main Dashboard Screenshot]**

**Upload Interface:**

- Drag-and-drop functionality for easy file selection
- File validation with immediate feedback
- Progress indicators during upload process
- Support for multiple image formats (PNG, JPEG, TIFF)

**[Space reserved for Figure 4.2: Upload Interface Screenshot]**

## 4.2 Back-End API Implementation

The core API infrastructure has been successfully implemented with the following capabilities:

**Image Processing Pipeline:**

- RESTful API endpoints using FastAPI
- Secure file upload handling with validation
- Integration with spatx_core library for model predictions
- Comprehensive error handling and logging

**Model Integration:**

- Successfully loaded pre-trained models (446MB CiT-Net model)
- Implemented prediction pipeline for breast cancer spatial transcriptomics
- Support for 51 gene expression predictions
- Batch processing capabilities for efficient computation

## 4.3 Technical Implementation Details

**Key Technical Achievements:**

1. **Successful Model Loading:** Resolved import path issues and successfully integrated the spatx_core library with the FastAPI backend.

2. **File Management System:** Implemented secure file upload, validation, and temporary storage mechanisms.

3. **API Architecture:** Created robust REST API endpoints with proper error handling and response formatting.

4. **Development Workflow:** Established efficient development and testing procedures.

**[Space reserved for Figure 4.3: Results Visualization Example]**

**[Space reserved for Figure 4.4: Backend Architecture Diagram]**

**Current Status:**

- Backend server successfully running on port 8002
- Frontend application built and ready for deployment
- Model predictions functioning correctly
- Full end-to-end pipeline operational

---

# Chapter 5: Conclusion and Future Work

## 5.1 Summary of Progress

This project aims to build a crucial bridge between advanced AI models and the wider biomedical community. In the first half of the semester, substantial progress has been achieved:

**Completed Milestones:**

- Comprehensive literature review and system design
- Full-stack application architecture implementation
- Successful integration of spatx_core library with FastAPI backend
- Development of responsive React frontend interface
- Complete image upload and processing pipeline
- Model loading and prediction functionality
- Error handling and user feedback systems

**Technical Achievements:**

- Resolution of complex import path and dependency issues
- Successful deployment on lab server environment
- Implementation of modern web technologies and best practices
- Creation of maintainable and scalable codebase

## 5.2 Future Work Plan

The plan for the remainder of the semester focuses on enhancement and deployment:

**Phase 1: Core Functionality Enhancement (Weeks 9-11)**

1. **Results Visualization Enhancement:** Develop advanced visualization components to display gene expression heatmaps overlaid on original H&E images
2. **Performance Optimization:** Implement caching mechanisms and optimize model inference time
3. **User Experience Improvements:** Add loading animations, progress tracking, and enhanced error messages

**Phase 2: Advanced Features (Weeks 12-14)**

1. **Data Management System:** Implement secure storage and retrieval of user submissions and predictions
2. **Batch Processing:** Enable processing of multiple images simultaneously
3. **Export Functionality:** Allow users to download results in various formats (CSV, JSON, images)

**Phase 3: Testing and Deployment (Weeks 15-16)**

1. **Comprehensive Testing:** Implement unit tests, integration tests, and user acceptance testing
2. **Security Audit:** Ensure secure file handling and data protection measures
3. **Production Deployment:** Deploy to cloud infrastructure for public accessibility
4. **Documentation:** Complete user guides and technical documentation

**Long-term Vision:**

- Integration of additional tissue types and prediction models
- Implementation of user authentication and project management features
- Development of collaborative features for research teams
- Integration with existing bioinformatics workflows and tools

---

# References

He, B., Chen, B., Tan, T., Qin, W., Wang, J., & Zhang, K. (2024). A visual–omics foundation model to bridge histopathology with spatial transcriptomics. _Nature Methods_, 21(3), 395-408.

[Note: Additional references to be added upon completion of literature review for HisToGene and ST-Net papers, along with other relevant works in spatial transcriptomics and computational pathology]

---

**Document prepared by:** Tanishk (2022BB11400)  
**Date:** November 2025  
**Project:** BBD451 B.Tech Major Project (BB1)  
**Supervisor:** Prof. Ishaan Gupta
