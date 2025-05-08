import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function currencyFormatter(
  value: number,
  currency: string
): string {
  const formatter = new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: currency,
  });
  return formatter.format(value);
}

export function numbersFormatter(value: number): string {
  const formatter = new Intl.NumberFormat("en-US");
  return formatter.format(value);
}