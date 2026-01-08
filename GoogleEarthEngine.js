// Define Changa Manga area

var changaManga = ee.Geometry.Rectangle([
  73.95, 31.05,  // lower-left
  74.10, 31.20   // upper-right
]);


// Load Sentinel-2 Level-2A collection

var s2 = ee.ImageCollection("COPERNICUS/S2_SR")
  .filterBounds(changaManga)
  .filterDate("2023-01-01", "2024-12-31")
  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40));  // relaxed for monthly


// NDVI function

var addNDVI = function(image) {
  var ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI");
  return image.addBands(ndvi).select("NDVI");
};

// Apply NDVI to collection
var ndviCollection = s2.map(addNDVI);


// Generate 24 monthly images and export each

for (var m = 0; m < 24; m++) {
  var start = ee.Date("2023-01-01").advance(m, "month");
  var end = start.advance(1, "month");

  var monthly_image = ndviCollection
    .filterDate(start, end)
    .median()
    .clip(changaManga);

  Export.image.toDrive({
    image: monthly_image,
    description: 'NDVI_' + start.format('YYYY_MM').getInfo(),
    scale: 10,
    region: changaManga,
    fileFormat: 'GeoTIFF',
    maxPixels: 1e13
  });
}


// Visualize first month

Map.centerObject(changaManga, 11);
Map.addLayer(
  ndviCollection.filterDate("2023-01-01", "2023-02-01").median(),
  {min: 0, max: 0.8, palette: ["brown", "yellow", "green"]},
  "NDVI Jan 2023"
);
