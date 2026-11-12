// Audio choisi par l'utilisateur, en attente de transcription. Il reste dans le navigateur.
export interface SelectedAudio {
  blob: Blob;
  url: string; // URL d'objet locale (blob:), pour le lecteur et le téléchargement
  filename: string;
  source: "micro" | "fichier";
  durationS: number | null; // inconnue tant que le navigateur ne l'a pas lue
}
