export const addMetersToLongitude = (longitude: number, meters: number) => {
  const earthRadius = 6378137; // Radius of the Earth in meters
  return longitude + (meters / earthRadius) * (180 / Math.PI);
};
