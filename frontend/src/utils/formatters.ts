/**
 * Format number to Indian currency format with K, L, C notation
 * K = Thousand (1,000)
 * L = Lakh (1,00,000)
 * Cr = Crore (1,00,00,000)
 * Shows exact values without additional rounding
 */
export const formatINR = (value: number, decimals: number = 2): string => {
  const absValue = Math.abs(value);
  const sign = value < 0 ? '-' : '';
  
  if (absValue >= 10000000) { // 1 Crore or more
    return `${sign}₹${(absValue / 10000000).toFixed(decimals)}Cr`;
  } else if (absValue >= 100000) { // 1 Lakh or more
    return `${sign}₹${(absValue / 100000).toFixed(decimals)}L`;
  } else if (absValue >= 1000) { // 1 Thousand or more
    return `${sign}₹${(absValue / 1000).toFixed(decimals)}K`;
  } else {
    return `${sign}₹${absValue.toFixed(decimals)}`;
  }
};

/**
 * Format number for chart price display (without ₹ symbol)
 */
export const formatChartPrice = (value: number): string => {
  const absValue = Math.abs(value);
  
  if (absValue >= 10000000) {
    return `${(value / 10000000).toFixed(2)}Cr`;
  } else if (absValue >= 100000) {
    return `${(value / 100000).toFixed(2)}L`;
  } else if (absValue >= 1000) {
    return `${(value / 1000).toFixed(2)}K`;
  } else {
    return value.toFixed(2);
  }
};
