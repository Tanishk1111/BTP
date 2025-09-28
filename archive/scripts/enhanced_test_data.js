// Enhanced test data with realistic gene expression patterns
const enhancedTestPredictions = [
  // 044x027_TENX99.png - Tissue region with moderate expression
  {
    barcode: "044x027",
    wsi_id: "TENX99", 
    x: 12963.550612,
    y: 20969.702956,
    image_path: "uploads/044x027_TENX99.png",
    ABCC11: 1.234,  // Medium expression
    ADH1B: 2.456,   // High expression  
    ADIPOQ: 0.789,  // Low expression
    ANKRD30A: 1.567,
    AQP1: 0.923,
    region: "Epithelial",
    confidence: 0.92
  },
  // 053x074_TENX99.png - Different expression profile
  {
    barcode: "053x074", 
    wsi_id: "TENX99",
    x: 15000,
    y: 25000,
    image_path: "uploads/053x074_TENX99.png", 
    ABCC11: 0.487,  // Low expression
    ADH1B: 1.543,   // Medium expression
    ADIPOQ: 2.234,  // High expression
    ANKRD30A: 0.634,
    AQP1: 1.876,
    region: "Stromal",
    confidence: 0.88
  },
  // 094x076_TENX99.png - High expression region
  {
    barcode: "094x076",
    wsi_id: "TENX99",
    x: 36022.374142, 
    y: 44499.114721,
    image_path: "uploads/094x076_TENX99.png",
    ABCC11: 2.123,  // High expression
    ADH1B: 0.654,   // Low expression
    ADIPOQ: 1.876,  // Medium-high expression
    ANKRD30A: 2.890,
    AQP1: 1.432,
    region: "Immune",
    confidence: 0.95
  },
  // 134x031_TENX99.png - Mixed expression
  {
    barcode: "134x031",
    wsi_id: "TENX99", 
    x: 25000,
    y: 30000,
    image_path: "uploads/134x031_TENX99.png",
    ABCC11: 1.567,  // Medium expression
    ADH1B: 2.890,   // High expression
    ADIPOQ: 0.432,  // Low expression
    ANKRD30A: 1.234,
    AQP1: 2.567,
    region: "Neural",
    confidence: 0.91
  },
  // 156x081_TENX99.png - Low expression region
  {
    barcode: "156x081",
    wsi_id: "TENX99",
    x: 38375.315318,
    y: 73675.585309, 
    image_path: "uploads/156x081_TENX99.png",
    ABCC11: 0.721,  // Low expression
    ADH1B: 1.988,   // Medium-high expression
    ADIPOQ: 2.654,  // High expression
    ANKRD30A: 0.543,
    AQP1: 1.234,
    region: "Lymphoid",
    confidence: 0.89
  }
];

// Additional synthetic data to make it more comprehensive
const generateAdditionalPredictions = () => {
  const additionalData = [];
  const regions = ["Epithelial", "Stromal", "Immune", "Neural", "Lymphoid", "Metabolic"];
  const genes = ["ABCC11", "ADH1B", "ADIPOQ", "ANKRD30A", "AQP1"];
  
  // Generate 20 more synthetic predictions for demonstration
  for (let i = 0; i < 20; i++) {
    const prediction = {
      barcode: `synthetic_${String(i).padStart(3, '0')}`,
      wsi_id: "TENX99",
      x: 10000 + Math.random() * 50000,
      y: 10000 + Math.random() * 60000,
      image_path: `uploads/044x027_TENX99.png`, // Use existing image as placeholder
      region: regions[Math.floor(Math.random() * regions.length)],
      confidence: 0.8 + Math.random() * 0.15
    };
    
    // Generate realistic gene expression values
    genes.forEach(gene => {
      // Create some correlation between genes and add realistic noise
      const baseExpression = Math.random() * 2.5;
      const noise = (Math.random() - 0.5) * 0.5;
      prediction[gene] = Math.max(0.001, baseExpression + noise);
    });
    
    additionalData.push(prediction);
  }
  
  return additionalData;
};

const allPredictions = [...enhancedTestPredictions, ...generateAdditionalPredictions()];
const selectedGenes = ["ABCC11", "ADH1B", "ADIPOQ", "ANKRD30A", "AQP1"];

// Store in localStorage
localStorage.setItem('predictionResults', JSON.stringify(allPredictions));
localStorage.setItem('selectedGenes', JSON.stringify(selectedGenes));

console.log('Enhanced test data with', allPredictions.length, 'predictions stored in localStorage');
console.log('Genes:', selectedGenes);
console.log('Sample prediction:', allPredictions[0]);