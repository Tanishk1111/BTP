// Test data to put in localStorage for testing image loading
const testPredictions = [
  {
    barcode: "044x027",
    wsi_id: "TENX99", 
    x: 12963.550612,
    y: 20969.702956,
    image_path: "uploads/044x027_TENX99.png",
    ABCC11: 1.234,
    ADH1B: 2.456,
    ADIPOQ: 0.789
  },
  {
    barcode: "053x074", 
    wsi_id: "TENX99",
    x: 15000,
    y: 25000,
    image_path: "uploads/053x074_TENX99.png", 
    ABCC11: 0.987,
    ADH1B: 1.543,
    ADIPOQ: 2.234
  },
  {
    barcode: "094x076",
    wsi_id: "TENX99",
    x: 36022.374142, 
    y: 44499.114721,
    image_path: "uploads/094x076_TENX99.png",
    ABCC11: 2.123,
    ADH1B: 0.654,
    ADIPOQ: 1.876
  },
  {
    barcode: "134x031",
    wsi_id: "TENX99", 
    x: 25000,
    y: 30000,
    image_path: "uploads/134x031_TENX99.png",
    ABCC11: 1.567,
    ADH1B: 2.890,
    ADIPOQ: 0.432
  },
  {
    barcode: "156x081",
    wsi_id: "TENX99",
    x: 38375.315318,
    y: 73675.585309, 
    image_path: "uploads/156x081_TENX99.png",
    ABCC11: 0.721,
    ADH1B: 1.988,
    ADIPOQ: 2.654
  }
];

const selectedGenes = ["ABCC11", "ADH1B", "ADIPOQ"];

// Store in localStorage
localStorage.setItem('predictionResults', JSON.stringify(testPredictions));
localStorage.setItem('selectedGenes', JSON.stringify(selectedGenes));

console.log('Test data stored in localStorage');
console.log('Navigate to /results to test image loading');