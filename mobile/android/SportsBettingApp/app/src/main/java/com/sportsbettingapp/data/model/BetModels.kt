package com.sportsbettingapp.data.model

import kotlinx.serialization.Serializable

@Serializable
data class Bet(
    val matchup: String,
    val market: String? = null,
    val pick: String? = null,
    val units: String? = null,
    val result: String? = null
)
