package com.sportsbettingapp.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.sportsbettingapp.data.model.RunRequest
import com.sportsbettingapp.data.repository.ApiRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

class RunViewModel(
    private val repository: ApiRepository = ApiRepository.create()
) : ViewModel() {
    private val _runState = MutableStateFlow<UiState<String>>(UiState.Idle)
    val runState: StateFlow<UiState<String>> = _runState

    fun runModel(sport: String, gameDate: String) {
        _runState.value = UiState.Loading
        viewModelScope.launch {
            try {
                val response = repository.createRun(
                    RunRequest(
                        sport = sport,
                        gameDate = gameDate
                    )
                )
                val runId = response.runId ?: response.id
                if (runId.isNullOrBlank()) {
                    _runState.value = UiState.Error("Missing run id in response")
                } else {
                    _runState.value = UiState.Success(runId)
                }
            } catch (exception: Exception) {
                _runState.value = UiState.Error(exception.localizedMessage ?: "Failed to run model")
            }
        }
    }

    fun reset() {
        _runState.value = UiState.Idle
    }
}
