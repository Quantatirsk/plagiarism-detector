import { clsx, type ClassValue } from "clsx"

// Simplified cn function without tailwind-merge
export function cn(...inputs: ClassValue[]) {
  return clsx(inputs)
}
