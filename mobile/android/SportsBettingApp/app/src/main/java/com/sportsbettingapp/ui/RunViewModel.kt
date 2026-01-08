package com.sportsbettingapp.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.sportsbettingapp.data.model.RunRequestDto
import com.sportsbettingapp.data.repository.ApiRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class RunViewModel(
    private val repository: ApiRepository = ApiRepository.create()
) : ViewModel() {
    private val _runState = MutableStateFlow<UiState<String>>(UiState.Idle)
    val runState: StateFlow<UiState<String>> = _runState

    fun runModel(sport: String, gameDate: String) {
        _runState.value = UiState.Loading
        viewModelScope.launch {
            try {
                val response = withContext(Dispatchers.IO) {
                    repository.createRun(
                        RunRequestDto(
                            sport = sport,
                            gameDate = gameDate
                        )
                    )
                }
                _runState.value = UiState.Success(response.runId)
            } catch (exception: Exception) {
                _runState.value = UiState.Error(exception.localizedMessage ?: "Failed to run model")
            }
        }
    }

    fun reset() {
        _runState.value = UiState.Idle
    }
}
