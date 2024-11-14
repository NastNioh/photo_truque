def automated_decision(truque, comment):
    """
    Prend une décision basée sur la probabilité que l'image soit truquée et le commentaire généré.
    
    Paramètres:
    - truque: Booléen indiquant si l'image est truquée.
    - comment: Commentaire expliquant l'anomalie ou la normalité de l'image.
    
    Retourne:
    - 'Accepter' si l'image semble normale.
    - 'Rejeter' si l'image semble truquée ou si des anomalies majeures sont détectées.
    """

    if truque:
        return "Rejeter"
    
    # Vérifie des indications claires de manipulation ou défaut dans le commentaire
    keywords_to_reject = ["anomalie majeure", "truqué", "modification", "altération"]
    for keyword in keywords_to_reject:
        if keyword in comment.lower():
            return "Rejeter"
    
    # Si aucune anomalie n'est détectée
    return "Accepter"
