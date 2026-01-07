package com.sportsbettingapp.data.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class Prediction(
    val home: String,
    val away: String,
    @SerialName("primary_recommendation") val primaryRecommendation: String? = null,
    @SerialName("confidence") val confidence: String? = null,
    @SerialName("value_tier") val valueTier: String? = null,
    val edges: Map<String, Double>? = null
)
